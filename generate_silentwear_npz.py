# generate_silentwear_npz.py
# Run this on your Mac:
#   python /Users/qiwenwu/ETH/Onnx4Deeploy/generate_silentwear_npz.py

import sys
sys.path.insert(0, "/Users/qiwenwu/ETH/Onnx4Deeploy")

import numpy as np
from onnx4deeploy.models.speechnet_exporter import SpeechNetExporter

# ── Step 1: generate inputs.npz and outputs.npz from real EMG data ────────
print("=== Generating npz from SilentWear dataset ===")

exporter = SpeechNetExporter(
    save_path="/Users/qiwenwu/ETH/Onnx4Deeploy/onnx/model/speechnet_train_silentwear"
)
exporter._config_overrides = {
    "dataset":     "silentwear",
    "data_path":   "/Users/qiwenwu/ETH/SilentWear_data/data_raw_and_filt",
    "subject":     "S01",
    "session":     3,
    "batch":       1,
    "condition":   "vocalized",
    "n_batches":   16,
    "time_steps":  700,
    "num_classes": 9,
}
exporter.config = exporter.load_config() 
exporter.create_training_test_data()

# ── Step 2: inspect inputs.npz ────────────────────────────────────────────
print("\n=== Inspecting inputs.npz ===")
inp = np.load("/Users/qiwenwu/ETH/Onnx4Deeploy/onnx/model/speechnet_train_silentwear/inputs.npz")
print("keys:", list(inp.keys()))
print("arr_0000 shape:", inp["arr_0000"].shape)
print("arr_0000 dtype:", inp["arr_0000"].dtype)
print("arr_0000 range: min={:.4f}  max={:.4f}".format(
    inp["arr_0000"].min(), inp["arr_0000"].max()))
print("arr_0001 (label):", inp["arr_0001"])
print("meta_n_batches:", inp["meta_n_batches"])
print("meta_data_size:", inp["meta_data_size"])

# ── Step 3: inspect outputs.npz ───────────────────────────────────────────
print("\n=== Inspecting outputs.npz ===")
out = np.load("/Users/qiwenwu/ETH/Onnx4Deeploy/onnx/model/speechnet_train_silentwear/outputs.npz")
print("keys:", list(out.keys()))
print("loss:", out["loss"])
print("loss range: min={:.4f}  max={:.4f}".format(
    out["loss"].min(), out["loss"].max()))

print("\n=== Done ===")