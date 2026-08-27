# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""SpeechNet Model Exporter — SilentWear EMG silent speech recognition.

Based on: Spacone et al., "SilentWear: an Ultra-Low Power Wearable System for
EMG-based Silent Speech Recognition", arXiv: 2603.02847.

Default configuration (paper, FP32):
  Input:  (1, 1, 14, 700)  — 1-ch, 14 EMG channels × 700 time samples (1.4 s @ 500 Hz)
  Classes: 9 (8 commands + rest)
  ~15 K parameters

The deployment variant uses AvgPool (not MaxPool) and omits Dropout for Deeploy
tiling/gradient compatibility. BatchNorm folds into Conv during ONNX export.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter


class SpeechNetExporter(BaseONNXExporter):
    """ONNX exporter for SpeechNet (SilentWear EMG silent speech)."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        super().__init__(save_path, config_file)
        self.model_config = {}

    # ------------------------------------------------------------------ #
    # Configuration                                                        #
    # ------------------------------------------------------------------ #

    def load_config(self) -> Dict[str, Any]:
        config = {
            "batch_size": 1,
            "num_channels": 14,  # EMG differential channels
            "time_steps": 700,  # 1.4 s @ 500 Hz
            "num_classes": 9,  # 8 commands + rest
            "opset_version": 17,
            # Training
            "training_strategy": "full",  # "full" | "last_layer" | "custom"
            "custom_trainable_params": [],
            "learning_rate": 0.001,
            "n_batches": 4,
            "n_accum": 1,
            "data_size": None,
            "pretrained_weights": None,  # path to .pt checkpoint
            "pretrained_key": "model_state_dict",  # key inside the checkpoint dict
            "data_path": "/app/SilentWear_data/data_raw_and_filt",
            "subject": "S01",
            "session": 3,
            "batch": 1,
            "condition": "vocalized",
            "stratified_sampling": False,
            # QW: zeroth-order (MeZO) config — perturbation scale + base seed for the ZO graphs. -- QW
            "zo": {"epsilon": 0.01, "seed": 42, "exceptions": []},
        }

        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)

        self.model_config = config
        return config

    # ------------------------------------------------------------------ #
    # Model factory                                                        #
    # ------------------------------------------------------------------ #

    def create_model(self) -> torch.nn.Module:
        from .pytorch_models.speechnet.speechnet import SpeechNetDeploy

        model = SpeechNetDeploy(
            num_channels=self.model_config["num_channels"],
            time_steps=self.model_config["time_steps"],
            num_classes=self.model_config["num_classes"],
        )
        ckpt_path = self.model_config.get("pretrained_weights")
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            key = self.model_config.get("pretrained_key")
            state_dict = ckpt[key] if (key and isinstance(ckpt, dict) and key in ckpt) else ckpt
            model.load_state_dict(state_dict, strict=True)
            print(f"  Loaded pretrained weights from {ckpt_path}")

        # QW: BN-fold fix for on-device head-only fine-tuning ----------------- QW
        # Fold BatchNorm into the preceding Conv when the feature extractor is frozen
        # (last_layer strategy) or explicitly requested.  Deeploy's training BatchNorm
        # kernel (BatchNormInternal) recomputes *batch* statistics; with on-device
        # batch-size-1 fine-tuning this normalises each window by its own spatial stats,
        # corrupting the features relative to inference (which uses running stats).
        # Folding removes the BN op so the frozen features match inference exactly,
        # which is mathematically identical in eval mode (zero-shot accuracy unchanged).
        strategy = self.model_config.get("training_strategy", "full")
        if self.model_config.get("fold_bn", False) or strategy == "last_layer":
            self._fold_bn_into_conv(model)
            print("  Folded BatchNorm → Conv (running-stat features; required for batch-1 on-device FT)")
        # QW: end BN-fold fix --------------------------------------------------- QW
        return model

    @staticmethod
    def _fold_bn_into_conv(model: torch.nn.Module) -> torch.nn.Module:  # QW: added by QW
        """Fold each block's BatchNorm2d into its preceding Conv2d and replace the
        BN with nn.Identity (exact for eval-mode BN; uses running_mean/running_var)."""
        import torch.nn as nn

        for blk in model.blocks:
            conv, bn = blk[0], blk[1]
            if not isinstance(bn, nn.BatchNorm2d):
                continue
            std = torch.sqrt(bn.running_var + bn.eps)
            scale = bn.weight / std  # (C,)
            conv.weight.data = conv.weight.data * scale.reshape(-1, 1, 1, 1)
            if conv.bias is None:
                conv.bias = nn.Parameter(torch.zeros(conv.weight.shape[0]))
            conv.bias.data = (conv.bias.data - bn.running_mean) * scale + bn.bias
            blk[1] = nn.Identity()
        return model

    def create_brevitas_model(self) -> torch.nn.Module:
        """QZO: INT8 Brevitas SpeechNet (conv+fc int8, BN unfolded fp32). Used by -mode q-zo-train."""
        import os
        from .pytorch_models.speechnet.speechnet_quant import QuantSpeechNetDeploy
        model = QuantSpeechNetDeploy(
            num_channels=self.model_config["num_channels"],
            time_steps=self.model_config["time_steps"],
            num_classes=self.model_config["num_classes"],
        )
        ckpt = self.model_config.get("pretrained_weights")
        if ckpt and os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(sd, dict):
                sd = sd.get(self.model_config.get("pretrained_key"), sd)
            remap = {}
            for k, v in sd.items():                     # SpeechNetDeploy blocks.N.0=conv/.1=bn -> .conv/.bn
                p = k.split(".")
                if k.startswith("blocks.") and len(p) >= 4 and p[2] in ("0", "1"):
                    remap[f"blocks.{p[1]}.{'conv' if p[2]=='0' else 'bn'}." + ".".join(p[3:])] = v
                else:
                    remap[k] = v
            missing, unexpected = model.load_state_dict(remap, strict=False)
            print(f"  QuantSpeechNet: loaded pretrained (remapped); missing={len(missing)} unexpected={len(unexpected)}")
        else:
            for n, param in model.named_parameters():
                if "weight" in n and param.dim() > 1:
                    torch.nn.init.normal_(param, 0.0, 0.05)
                if "bias" in n:
                    torch.nn.init.uniform_(param, 0.01, 0.02)
            print("  QuantSpeechNet: no pretrained_weights -> random init")
        return model

    def get_calibration_data(self):
        """Representative windows for PTQ activation-scale calibration (real SilentWear if available)."""
        import numpy as np
        n = int(self.config.get("calib_samples", 8))
        shape = (n, 1, self.model_config["num_channels"], self.model_config["time_steps"])
        if self.config.get("dataset", "random") == "silentwear":
            try:
                ds = self.get_data_source()
                X, _ = ds.load_batches(n, shape[1:], self.model_config["num_classes"], seed=42)
                return np.asarray(X[:n], np.float32).reshape(shape)
            except Exception as e:
                print(f"  calibration: silentwear unavailable ({e}); using random")
        return np.random.randn(*shape).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Shape helpers                                                        #
    # ------------------------------------------------------------------ #

    def get_input_shape(self) -> Tuple[int, ...]:
        return (
            self.config["batch_size"],
            1,  # single input channel
            self.config["num_channels"],
            self.config["time_steps"],
        )

    def _get_config_string(self) -> str:
        return (
            f"_speechnet_{self.config['num_channels']}ch"
            f"_{self.config['time_steps']}t"
            f"_{self.config['num_classes']}cls"
        )

    # ------------------------------------------------------------------ #
    # Training strategy                                                   #
    # ------------------------------------------------------------------ #

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Pattern-based trainable parameter selection.

        Strategies:
        - "full":       Train all parameters (default).
        - "last_layer": Only the FC classifier.
        - "custom":     Explicit list from config["custom_trainable_params"].
        """
        strategy = self.config.get("training_strategy", "full")

        _FREEZE = {
            "full": lambda n: False,
            "last_layer": lambda n: "fc" not in n,
            "custom": lambda n: n not in self.config.get("custom_trainable_params", []),
        }

        if strategy not in _FREEZE:
            print(f"  Unknown strategy '{strategy}', using 'full'")
            strategy = "full"

        requires_grad = [n for n in all_param_names if not _FREEZE[strategy](n)]
        frozen = [n for n in all_param_names if _FREEZE[strategy](n)]

        print(f"\n  Training Strategy: '{strategy}'")
        print(
            f"   Total: {len(all_param_names)}  Trainable: {len(requires_grad)}  Frozen: {len(frozen)}"
        )
        if frozen:
            print(f"   Frozen: {frozen[:5]}{'...' if len(frozen) > 5 else ''}")
        return requires_grad

    # ------------------------------------------------------------------ #
    # Return the data source for training mini-batch generation          #
    # ------------------------------------------------------------------ #
    def get_data_source(self):
        dataset = self.config.get("dataset", "random")
        if dataset == "silentwear":
            from ..data.silent_wear_datasource import SilentWearDataSource

            cfg = self.config
            return SilentWearDataSource(
                data_path=cfg["data_path"],
                subject=cfg.get("subject", "S01"),
                session=cfg.get("session", 1),
                batch=cfg.get("batch", 1),
                condition=cfg.get("condition", "vocalized"),
                window_samples=cfg.get("time_steps", 700),
                downsample_rest=True,
                stratified_split=cfg.get("stratified_sampling", False),
            )
        from ..data.random_datasource import RandomDataSource

        return RandomDataSource()

    # ------------------------------------------------------------------ #
    # Inference test data                                                  #
    # ------------------------------------------------------------------ #

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        if self.config.get("dataset", "random") == "silentwear":
            self._save_silentwear_eval_data(model, save_dir)
            return

        print("  Saving inference test data...")
        input_shape = self.get_input_shape()
        test_input = np.random.randn(*input_shape).astype(np.float32)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            test_output = model(torch.from_numpy(test_input)).numpy()
        if was_training:
            model.train()

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)
        print(f"   Input: {test_input.shape}  Output: {test_output.shape}")

    def _save_silentwear_eval_data(self, model: torch.nn.Module, save_dir: str):
        print("  Saving SilentWear inference test data...")
        from ..data.silent_wear_datasource import SilentWearDataSource

        cfg = self.config
        ds = SilentWearDataSource(
            data_path=cfg["data_path"],
            subject=cfg.get("subject", "S01"),
            session=cfg.get("session", 3),
            batch=cfg.get("batch", 1),
            condition=cfg.get("condition", "vocalized"),
            window_samples=cfg.get("time_steps", 700),
            downsample_rest=True,
        )
        inputs, labels = ds._load_windows()

        # --- run model on all windows, save full batch ---
        model.eval()
        all_outputs = []
        with torch.no_grad():
            for x in inputs:
                all_outputs.append(model(torch.from_numpy(x)).numpy())

        all_inputs_np = np.concatenate(inputs, axis=0)  # (N, 1, 14, 700)
        all_outputs_np = np.concatenate(all_outputs, axis=0)  # (N, 9)
        labels_np = np.concatenate(labels, axis=0)  # (N,)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(save_path / "inputs.npz", input=all_inputs_np, label=labels_np)
        np.savez(save_path / "outputs.npz", output=all_outputs_np)
        print(
            f"   Inputs: {all_inputs_np.shape}  Outputs: {all_outputs_np.shape}  ({len(inputs)} windows)"
        )
        preds = np.argmax(np.concatenate(all_outputs, axis=0), axis=-1)
        labels_arr = np.concatenate(labels, axis=0)
        classes = np.unique(labels_arr)
        per_class_recall = np.array([np.mean(preds[labels_arr == c] == c) for c in classes])
        balanced_acc = per_class_recall.mean()
        print(
            f"   Balanced accuracy = {balanced_acc:.4f} ({balanced_acc * 100:.1f}%)  ({len(inputs)} windows)"
        )
        for c, r in zip(classes, per_class_recall):
            print(f"     class {int(c)}: recall = {r:.3f}")

    # ------------------------------------------------------------------ #
    # Training test data                                                   #
    # ------------------------------------------------------------------ #

    def _frozen_pytorch_reference(
        self, init_map, test_inputs, labels_list, n_steps, n_accum,
        effective_data_size, learning_rate, grad_tensor_map,
    ):
        """exp1 / Option A — compute the training reference in PyTorch with FROZEN BN.

        ORT's ``BatchNormInternal(training_mode=1)`` normalises with live per-batch statistics, but the
        device trains with the FROZEN pretrained running stats (``BN_FROZEN_STATS``). Running the same
        loop in PyTorch with ``model.eval()`` (frozen BN, γ/β still trainable) reproduces exactly what the
        device computes, so the runner's bit-exact loss check becomes meaningful.

        Mirrors the device loop EXACTLY: same window order (``mb % data_size``), SUM gradient accumulation
        (plain ``SGD(lr)`` → ``w ← w − lr·Σ gradᵢ``), one optimiser step per ``n_accum`` mini-batches.
        Returns ``(all_losses, updated_weights)`` — ``updated_weights`` is ``init_map`` with the trainable
        tensors replaced by their post-training values (frozen tensors unchanged).
        """
        import torch

        model = self.create_model()          # SpeechNetDeploy; BN NOT folded for the 'full' strategy
        # Load the exact fixture initial weights (init_map) so the reference starts where the device does.
        sd = model.state_dict()
        for tname in list(sd.keys()):
            oname = tname.replace(".", "_")
            if oname in init_map:
                sd[tname] = torch.from_numpy(np.asarray(init_map[oname])).float().reshape(sd[tname].shape)
        model.load_state_dict(sd)
        model.eval()                          # FROZEN BN: running stats used and NOT updated; dropout off

        trainable = set(grad_tensor_map.keys())   # params that have a grad-accumulation buffer
        for tname, p in model.named_parameters():
            p.requires_grad_(tname.replace(".", "_") in trainable)
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=learning_rate)   # no momentum / no weight-decay → matches device
        crit = torch.nn.CrossEntropyLoss()                # mean; batch-1 ⇒ single-window loss

        all_losses = []
        for update_step in range(n_steps):
            opt.zero_grad()
            for accum_step in range(n_accum):
                mb = update_step * n_accum + accum_step
                x = torch.from_numpy(np.asarray(test_inputs[mb % effective_data_size])).float()
                y = torch.from_numpy(
                    np.atleast_1d(np.asarray(labels_list[mb % effective_data_size])).reshape(-1)
                ).long()
                loss = crit(model(x), y)
                loss.backward()               # SUM-accumulate into .grad (no zero between accum steps)
                all_losses.append(float(loss.detach().item()))
            opt.step()                        # w ← w − lr · Σ gradᵢ

        updated = {k: np.asarray(v).copy() for k, v in init_map.items()}
        with torch.no_grad():
            for tname, p in model.named_parameters():
                oname = tname.replace(".", "_")
                if oname in updated:
                    updated[oname] = p.detach().cpu().numpy().reshape(updated[oname].shape)
        return all_losses, updated

    def create_training_test_data(
        self, n_batches: int = None, num_data_inputs: int = 2, n_accum: int = None
    ) -> None:
        """
        Save inputs.npz / outputs.npz for training-mode validation.

        Inputs are random EMG-shaped float tensors; labels are random int64 class indices.
        """
        import onnx
        import onnxruntime as ort

        if n_batches is None:
            n_batches = self.config.get("n_batches", 4)
        if n_accum is None:
            n_accum = int(self.config.get("n_accum", 1))
        if n_batches % n_accum != 0:
            n_batches = max((n_batches // n_accum) * n_accum, n_accum)
            print(f"   n_batches adjusted to {n_batches} (must be divisible by n_accum={n_accum})")
        n_steps = n_batches // n_accum

        save_dir = Path(self.paths["output_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        input_shape = self.get_input_shape()
        num_classes = self.config.get("num_classes", 9)
        learning_rate = float(self.config.get("learning_rate", 0.001))

        print(
            f"   Training sim: n_batches={n_batches}  n_accum={n_accum}  n_steps={n_steps}  lr={learning_rate}"
        )

        _data_size_cfg = self.config.get("data_size", None)
        effective_data_size = (
            int(_data_size_cfg)
            if (_data_size_cfg and int(_data_size_cfg) < n_batches)
            else n_batches
        )

        # rng = np.random.default_rng(42)
        # test_inputs = [
        #     rng.standard_normal(input_shape).astype(np.float32) for _ in range(effective_data_size)
        # ]
        # labels_list = [
        #     rng.integers(0, num_classes, size=(input_shape[0],)).astype(np.int64)
        #     for _ in range(effective_data_size)
        # ]

        # Use real data set from SilentWear for training data
        data_source = self.get_data_source()
        test_inputs, labels_list = data_source.load_batches(
            effective_data_size, input_shape, num_classes, seed=42
        )

        init_map: dict = self._load_init_map(self.paths["network_infer"])

        train_model_onnx = onnx.load(self.paths["network_train"])
        grad_tensor_map: dict = {}
        for node in train_model_onnx.graph.node:
            if "InPlaceAccumulator" in node.op_type and len(node.input) >= 2:
                grad_tensor_name = node.input[1]
                if grad_tensor_name.endswith("_grad"):
                    grad_tensor_map[grad_tensor_name[:-5]] = grad_tensor_name

        for grad_name in grad_tensor_map.values():
            vi = onnx.helper.make_tensor_value_info(grad_name, onnx.TensorProto.FLOAT, None)
            train_model_onnx.graph.output.append(vi)

        session = ort.InferenceSession(
            train_model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        session_output_names = [o.name for o in session.get_outputs()]
        print(f"   Training model inputs:  {[i.name for i in session.get_inputs()]}")
        print(f"   Training model outputs: {session_output_names}")

        current_weights = {k: v.copy() for k, v in init_map.items()}
        all_losses: list = []
        feed_mb0: dict = {}

        _bn_frozen = bool(self.config.get("bn_frozen_stats", False))
        if _bn_frozen:
            # exp1 / Option A — frozen-BN reference in PyTorch (model.eval()); mirrors the device loop
            # exactly (window order, SUM accumulation, plain SGD). See experiments/exp1/PLAN.md.
            all_losses, current_weights = self._frozen_pytorch_reference(
                init_map, test_inputs, labels_list, n_steps, n_accum,
                effective_data_size, learning_rate, grad_tensor_map,
            )
            feed_mb0 = {
                k: (v.copy() if hasattr(v, "copy") else v)
                for k, v in self._build_input_feed(
                    session,
                    param_values={kk: vv.copy() for kk, vv in init_map.items()},
                    test_input=test_inputs[0],
                    labels=labels_list[0],
                    lazy_reset_grad=True,
                ).items()
            }
            print(f"   [exp1/Option A] frozen-BN PyTorch reference — {len(all_losses)} losses "
                  f"(first 5: {[round(x, 5) for x in all_losses[:5]]})")
        else:
            for update_step in range(n_steps):
                accumulated_grads = {
                    pname: np.zeros_like(current_weights[pname])
                    for pname in grad_tensor_map
                    if pname in current_weights
                }

                for accum_step in range(n_accum):
                    mb = update_step * n_accum + accum_step

                    feed = self._build_input_feed(
                        session,
                        param_values=current_weights,
                        test_input=test_inputs[mb % effective_data_size],
                        labels=labels_list[mb % effective_data_size],
                        lazy_reset_grad=(accum_step == 0),
                    )

                    if mb == 0:
                        feed_mb0 = {k: v.copy() if hasattr(v, "copy") else v for k, v in feed.items()}

                    raw_outputs = session.run(None, feed)
                    outputs_raw = dict(zip(session_output_names, raw_outputs))

                    for out_name, out_val in outputs_raw.items():
                        if "loss" in out_name.lower() and "grad" not in out_name.lower():
                            all_losses.append(float(np.array(out_val).flatten()[0]))
                            break

                    for pname, grad_name in grad_tensor_map.items():
                        if grad_name in outputs_raw and pname in accumulated_grads:
                            accumulated_grads[pname] += outputs_raw[grad_name]

                for pname, acc_grad in accumulated_grads.items():
                    current_weights[pname] -= learning_rate * acc_grad

        outputs_dict: dict = {k: v for k, v in current_weights.items()}
        outputs_dict["loss"] = np.array(all_losses, dtype=np.float32)
        print(f"   Reference losses ({'frozen-PyTorch' if _bn_frozen else 'ORT'}): "
              f"{[round(x, 5) for x in all_losses[:8]]}{'...' if len(all_losses) > 8 else ''}")

        final_model = onnx.load(self.paths["network"])
        final_input_names = [inp.name for inp in final_model.graph.input]
        grad_acc_names = {n for n in final_input_names if self._GRAD_ACC_SUFFIX in n}
        non_grad_names = [n for n in final_input_names if n not in grad_acc_names]

        save_dict: dict = {}
        for npz_idx, name in enumerate(non_grad_names):
            if name in feed_mb0:
                save_dict[f"arr_{npz_idx:04d}"] = feed_mb0[name]
            else:
                print(f"   non-grad input '{name}' not found in feed -- skipping")

        session_type: dict = {inp.name: inp.type for inp in session.get_inputs()}
        data_names = non_grad_names[:num_data_inputs]
        for mb in range(1, effective_data_size):
            for buf_idx, data_name in enumerate(data_names):
                inp_type = session_type.get(data_name, "tensor(float)")
                if inp_type == "tensor(int64)":
                    save_dict[f"mb{mb}_arr_{buf_idx:04d}"] = labels_list[mb]
                else:
                    save_dict[f"mb{mb}_arr_{buf_idx:04d}"] = test_inputs[mb]

        save_dict["meta_data_size"] = np.array([effective_data_size], dtype=np.int32)
        save_dict["meta_n_batches"] = np.array([n_batches], dtype=np.int32)
        save_dict["meta_n_accum"] = np.array([n_accum], dtype=np.int32)
        np.savez(save_dir / "inputs.npz", **save_dict)

        n_params = sum(1 for n in non_grad_names if n in init_map)
        n_grad = len(grad_acc_names)
        print(
            f"   inputs.npz: {len(non_grad_names)} base tensors "
            f"(data + {n_params} params; {n_grad} grad-acc-buf(s) omitted) "
            f"+ {(effective_data_size - 1) * num_data_inputs} DATA entries"
        )

        np.savez(save_dir / "outputs.npz", **outputs_dict)
        n_updated = sum(1 for k in outputs_dict if k in init_map)
        print(f"   outputs.npz: {len(outputs_dict)} tensors ({n_updated} updated params + loss)")

    # ------------------------------------------------------------------ #
    # QW: Zeroth-order (MeZO) multi-step training fixture                  #
    # ------------------------------------------------------------------ #
    def _zo_pytorch_reference(
        self, init_map, test_inputs, labels_list, n_steps, n_accum,
        effective_data_size, lr, eps, seed, q, node_id_map,
    ):
        """QW: ZO analogue of `_frozen_pytorch_reference` — the FULL N-step faithful MeZO run in PyTorch.

        Mirrors the device ZO loop: FROZEN BN (``model.eval()``, γ/β still perturbed), one SHARED
        perturbation direction ``z`` per update window (device Rademacher RNG ``_perturb_rademacher`` keyed
        by the per-param ``node_id`` from the graph — now consistent between zo_train/zo_update), scalar
        accumulation of ``(L₊ − L₋)`` over ``n_accum`` windows, and one in-place update
        ``θ ← θ − lr·g_proj·z`` per window; ``q`` directions averaged. Per-window seed convention:
        ``g = seed + update_step·q + q_i`` (the on-device runner must override the graph seed the same way).

        Returns ``(all_log_prob, all_lplus, all_lminus, updated_weights)``.
        """
        import torch
        from ..utils.onnx_node_implementations import _perturb_rademacher

        model = self.create_model()
        sd = model.state_dict()
        for tname in list(sd.keys()):
            oname = tname.replace(".", "_")
            if oname in init_map:
                sd[tname] = torch.from_numpy(np.asarray(init_map[oname])).float().reshape(sd[tname].shape)
        model.load_state_dict(sd)
        model.eval()                                  # FROZEN BN (running stats fixed; γ/β perturbed)

        # Trainable = the perturb targets (from zo_train); attach each torch param + its node_id.
        targets = []  # (oname, torch_param, node_id)
        for tname, p in model.named_parameters():
            oname = tname.replace(".", "_")
            if oname in node_id_map and node_id_map[oname] is not None:
                targets.append((oname, p, int(node_id_map[oname])))

        crit = torch.nn.CrossEntropyLoss()

        def z_for(param, gseed, node_id):
            z = _perturb_rademacher(np.zeros(tuple(param.shape), np.float32), int(gseed), int(node_id), 1.0, 1)
            return torch.from_numpy(np.ascontiguousarray(z)).reshape(param.shape)

        def add_(zs, coeff):
            if coeff == 0.0:
                return
            for (_, p, _), z in zip(targets, zs):
                p.add_(z, alpha=float(coeff))

        all_lp, all_lplus, all_lminus = [], [], []
        with torch.no_grad():
            for update_step in range(n_steps):
                for q_i in range(q):
                    gseed = int(seed) + update_step * q + q_i
                    zs = [z_for(p, gseed, nid) for (_, p, nid) in targets]   # SHARED z across the window
                    lp_sum = lm_sum = 0.0
                    for accum_step in range(n_accum):
                        mb = update_step * n_accum + accum_step
                        x = torch.from_numpy(np.asarray(test_inputs[mb % effective_data_size])).float()
                        y = torch.from_numpy(
                            np.atleast_1d(np.asarray(labels_list[mb % effective_data_size])).reshape(-1)
                        ).long()
                        add_(zs, +eps)                                       # +ε
                        logits_p = model(x)
                        lp = torch.log_softmax(logits_p, dim=-1).cpu().numpy().astype(np.float32)
                        Lp = float(crit(logits_p, y).item())
                        add_(zs, -2.0 * eps)                                 # −ε
                        Lm = float(crit(model(x), y).item())
                        add_(zs, +eps)                                       # restore θ
                        lp_sum += Lp; lm_sum += Lm
                        all_lp.append(lp); all_lplus.append(Lp); all_lminus.append(Lm)
                    g_proj = (lp_sum - lm_sum) / (2.0 * eps * n_accum)
                    add_(zs, -lr * g_proj / q)                               # in-place update along z

        updated = {k: np.asarray(v).copy() for k, v in init_map.items()}
        for (oname, p, _) in targets:
            updated[oname] = p.detach().cpu().numpy().reshape(updated[oname].shape)
        return all_lp, all_lplus, all_lminus, updated

    def create_training_test_data_zo(
        self, n_batches: int = None, num_data_inputs: int = 2, n_accum: int = None
    ) -> None:
        """QW: ZO multi-step fixture — the ZO analogue of `create_training_test_data`.

        Simulates the full MeZO run (via `_zo_pytorch_reference`) and packs the SAME format as the BP
        fixture so the on-device ZO runner can validate per-step + final:
          inputs.npz  : mb0 feed (arr_XXXX = window₀, label₀) + every other window/label (mb{mb}_arr_)
                        + meta (data_size, n_batches, n_accum) + ZO meta (eps, seed, lr, q).
          outputs.npz : final trained weights + per-step `log_prob` (device log-softmax output)
                        + scalar `loss_plus` / `loss_minus` (for g_proj validation).
        """
        import numpy as np
        import onnx

        if n_batches is None:
            n_batches = self.config.get("n_batches", 4)
        if n_accum is None:
            n_accum = int(self.config.get("n_accum", 1))
        if n_batches % n_accum != 0:
            n_batches = max((n_batches // n_accum) * n_accum, n_accum)
        n_steps = n_batches // n_accum

        save_dir = Path(self.paths["output_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        input_shape = self.get_input_shape()
        num_classes = self.config.get("num_classes", 9)
        lr = float(self.config.get("learning_rate", 0.001))
        zo = self.config.get("zo", {})
        eps = float(zo.get("epsilon", 0.01)); seed = int(zo.get("seed", 42)); q = int(zo.get("q", 1))
        _dc = self.config.get("data_size", None)
        effective_data_size = int(_dc) if (_dc and int(_dc) < n_batches) else n_batches

        print(f"   ZO training sim: n_batches={n_batches} n_accum={n_accum} n_steps={n_steps} "
              f"q={q} lr={lr} eps={eps} seed={seed}")

        data_source = self.get_data_source()
        test_inputs, labels_list = data_source.load_batches(
            effective_data_size, input_shape, num_classes, seed=42
        )
        init_map = self._load_init_map(self.paths["network_infer"])

        # Per-param node_id from zo_train (== zo_update after the idx-consistency fix).
        zt = onnx.load(self.paths["network_zo_train"]).graph
        node_id_map = {
            n.input[0]: next((a.i for a in n.attribute if a.name == "idx"), None)
            for n in zt.node if "Perturb" in n.op_type
        }

        all_lp, all_lplus, all_lminus, updated = self._zo_pytorch_reference(
            init_map, test_inputs, labels_list, n_steps, n_accum,
            effective_data_size, lr, eps, seed, q, node_id_map,
        )

        # QW: Step 6 — outputs.npz carries ONLY the 22 trainable weights, plus log_prob + loss_plus +
        #     loss_minus. Frozen BN running stats are NOT emitted (they never change under the ZO update).
        #     Source the trainable names from the Perturb-node base weights (node_id_map keys) — these are
        #     the 22 trainable params whether zo_train carries them as INITIALIZERS (reference design) or
        #     as inputs. (Previously read from zt.input, which is empty in the initializer form.) -- QW
        trainable_names = list(node_id_map.keys())  # -- QW
        outputs_dict = {k: updated[k] for k in trainable_names if k in updated}  # -- QW
        outputs_dict["log_prob"] = np.concatenate(all_lp, axis=0).astype(np.float32)
        outputs_dict["loss_plus"] = np.array(all_lplus, dtype=np.float32)
        outputs_dict["loss_minus"] = np.array(all_lminus, dtype=np.float32)
        np.savez(save_dir / "outputs.npz", **outputs_dict)
        n_upd = sum(1 for k in outputs_dict if k in init_map)
        print(f"   outputs.npz: {n_upd} final weights + log_prob {outputs_dict['log_prob'].shape} "
              f"+ loss_plus/minus ({len(all_lplus)} step-losses)")

        # QW: Step 5 — inputs.npz packs ALL zo_train graph inputs in graph-input order:
        #     `input`, `label`, + the 22 trainable weights (now graph INPUTS, not initializers), so
        #     `testInitWeights` exist for the device runner. Mirrors the BP `create_training_test_data`
        #     packing. Non-data windows still carry only data/label (mb{mb}_arr_0000/0001). -- QW
        zt_input_names = [i.name for i in zt.input]        # ['input', 'label', + 22 weights]
        feed0 = {
            "input": np.asarray(test_inputs[0], np.float32),
            "label": np.atleast_1d(np.asarray(labels_list[0])).reshape(-1, 1).astype(np.int64),
        }
        for wname, wval in init_map.items():  # -- QW: add the trainable weights to the mb0 feed -- QW
            feed0[wname] = np.asarray(wval)  # -- QW
        save_dict = {}
        for npz_idx, name in enumerate(zt_input_names):
            if name in feed0:
                save_dict[f"arr_{npz_idx:04d}"] = feed0[name]
            else:  # -- QW
                print(f"   zo_train input '{name}' missing from feed0 -- skipping")  # -- QW
        for mb in range(1, effective_data_size):
            save_dict[f"mb{mb}_arr_0000"] = np.asarray(test_inputs[mb], np.float32)
            save_dict[f"mb{mb}_arr_0001"] = np.atleast_1d(
                np.asarray(labels_list[mb])).reshape(-1, 1).astype(np.int64)
        save_dict["meta_data_size"] = np.array([effective_data_size], np.int32)
        save_dict["meta_n_batches"] = np.array([n_batches], np.int32)
        save_dict["meta_n_accum"] = np.array([n_accum], np.int32)
        save_dict["meta_zo_eps"] = np.array([eps], np.float32)
        save_dict["meta_zo_seed"] = np.array([seed], np.int32)
        save_dict["meta_zo_lr"] = np.array([lr], np.float32)
        save_dict["meta_zo_q"] = np.array([q], np.int32)
        np.savez(save_dir / "inputs.npz", **save_dict)
        print(f"   inputs.npz: {len(zt_input_names)} base tensors + "
              f"{(effective_data_size - 1) * num_data_inputs} DATA entries + meta")
