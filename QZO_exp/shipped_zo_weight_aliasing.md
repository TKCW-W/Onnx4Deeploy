# Shipped Deeploy QMCUNetZO — How ZO weight UPDATES mutate graph *initializers* in place

Investigation of the SHIPPED (read-only) Deeploy repo `/Users/qiwenwu/ETH/Deeploy`, fixture
`DeeployTest/Tests/Models/QMCUNetZO/`. All source citations are `file:line`. Nothing was modified.

---

## TL;DR — answer to the puzzle

- **Are the "initializers" really constant?** No. In Deeploy an ONNX *initializer* is imported as a
  `ConstantBuffer`, but a `ConstantBuffer` is emitted as a **plain, writable `static` C array pre-loaded
  with the weight values** — *not* a `const` array. The `const` variant is explicitly commented out in the
  template. So the weight lives in mutable L2 RAM; "initializer" only means "has a compile-time initial value."
- **How are they updated in place?** The ZO update op `RQSPerturbRademacher` is emitted with its
  **ONNX input name == output name == the weight tensor name** (e.g. `node_conv2d_weight` → `node_conv2d_weight`).
  Because the buffer name is identical, the generated kernel reads and writes the *same* C symbol / same L2
  address: `ApplyPerturbQuantRademacher_CHW(&data_in[...], &data_out[...], ...)` with `data_in == data_out`.
  That is a genuine in-place mutation of the weight array.
- **How do the train and update graphs alias the same buffer?** Two mechanisms depending on layout:
  1. **Initializer layout (the QMCUNetZO fixture itself):** the weight is a graph *initializer* in **both**
     `network.onnx` (train) and `network_zo_update.onnx` (update). Each network is code-generated to a
     `static PI_L2 … node_conv2d_weight[…]` array. Aliasing is by **identical global C symbol name across the
     two translation units** — the update TU declares the symbol `extern` and links against the train TU's
     definition, so both networks touch the same L2 address. No copying.
  2. **Input layout (the BP/SGD `deeploytraintest.c` path, and our vendored ZO path):** weights are graph
     *inputs* (`DeeployNetwork_inputs[]`) and the two networks are wired either by explicit `memcpy` each step
     (shipped SGD harness) or by pointer redirection (`_patch_shared_buffers`, vendored ZO).

The QMCUNetZO fixture is the **initializer + same-name-in/out + extern-symbol** design.

---

## 1. Graph structure findings

The fixture ships **two graph sets**:

```
Tests/Models/QMCUNetZO/network.onnx                     (single combined graph, top level)
Tests/Models/QMCUNetZO/QMCUNetZO/network.onnx           (train graph)
Tests/Models/QMCUNetZO/QMCUNetZO/network_zo_update.onnx (update graph)   <-- the ZO update
```

Loaded with `onnx` + `collections.Counter` (onnx 1.22.0):

### `QMCUNetZO/network.onnx` — the ZO TRAIN graph
- inputs = 2: `['input', 'label']`
- outputs = 1: `['log_prob']`
- initializers = **336**
- op_types:
  `Div:30, Add:30, Round:30, Clip:58, Sub:30, Mul:30, RQSPerturbRademacher:84, Conv:42, RequantShift:42, ReduceMean:1, PerturbRademacher:2, Gemm:1, SoftmaxCrossEntropyLoss:1`
- The trainable weights (e.g. `node_conv2d_weight`, `node_conv2d_1_weight`, …) are **`graph.initializer`**,
  NOT `graph.input`. Only `input` and `label` are graph inputs. → each forward perturbs the weight
  initializers (84 RQSPerturbRademacher = 2×42, i.e. perturb + un-perturb around the loss), computes the loss.

### `QMCUNetZO/network_zo_update.onnx` — the ZO UPDATE graph
- inputs = **0**, outputs = **0**, initializers = **44**
- op_types: `RQSPerturbRademacher: 44` (nothing else)
- Every `RQSPerturbRademacher` node has **input[0] == output[0] == the weight initializer name**:

  ```
  NODE: RQSPerturbRademacher  name=rqs_perturb_rademacher_node_conv2d_weight
    inputs : ['node_conv2d_weight', 'node_conv2d_weight_mul']
    outputs: ['node_conv2d_weight']          <-- SAME NAME as input[0] => in-place
    attr: div=32768  idx=0  n_levels=256  seed=42  signed=1
  ```
  `init0: node_conv2d_weight dims [16,3,3,3] dtype 1(float)`.
- Cross-check: **all 44 update-graph weight initializer names are also initializers in the train graph**
  (`update-init names ALL present in train-init? True`, shared count 44). Same names ⇒ (after codegen) same
  global C symbols ⇒ same physical buffer.

So a ZO step = *(train graph)* two perturbed forwards to get L+ and L−, then *(update graph)* one pass of 44
in-place `RQSPerturbRademacher` ops that add `-lr·g_proj·z` back onto the very same weight arrays.

---

## 2. How an "initializer" becomes a MUTABLE buffer (the crux)

### 2a. Import: initializer → `ConstantBuffer`
- Graph *inputs* become `VariableBuffer(is_input=True)`:
  `Deeploy/DeeployTypes.py:2505` `_createIOBindings` → `2511 nb = ctxt.VariableBuffer(...)`, `2512 nb.is_input = True`.
- Graph *initializers / constants* become `ConstantBuffer` via `hoistConstant`:
  `Deeploy/DeeployTypes.py:962 def hoistConstant(...)`, called from `1117-1118`
  (`if type(inputNode) == gs.ir.tensor.Constant … ctxt.hoistConstant(inputNode)`).
- `ConstantBuffer` (`Deeploy/DeeployTypes.py:399`) is "compile-time constant … always live" (`411 self._live = True`).

### 2b. Emit: `ConstantBuffer` is a WRITABLE `static` array, NOT `const`
`Deeploy/Targets/PULPOpen/Templates/AllocateTemplate.py`:
```py
17  pulpL2GlobalInitTemplate = NodeTemplate(
18      "static PI_L2 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n")
...
23  #pulpL2GlobalInitTemplate = NodeTemplate("static const ${type} ${name}[${size}];\n")   # <-- const variant DISABLED
...
49  % elif _memoryLevel == "L2" or _memoryLevel is None:
50      static PI_L2 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
```
The emitted symbol is `static PI_L2 float32_t node_conv2d_weight[432] = { …init values… };` — a **non-const,
writable** L2 array seeded with the exported weights. The line-23 `const` version is intentionally commented
out. **So "initializer" ⇒ "writable array with an initial value", not "read-only constant".** This is the whole
trick: nothing needs to "promote" the buffer — a Deeploy `ConstantBuffer` is already mutable memory.

### 2c. In-place write by the kernel
`Deeploy/Targets/PULPOpen/Templates/RQSPerturbRademacherTemplate.py:46-53`:
```c
ApplyPerturbQuantRademacher_CHW((const int8_t *) &${data_in}[${nodeName}_chunk_start],
                                (int8_t *)       &${data_out}[${nodeName}_chunk_start],
                                (const int32_t *) ${mul}, ${log2Dstring}, ${channel_width},
                                chunk_seed, ${nodeName}_local_size, ${nodeName}_chunk_start);
```
The parser sets `data_in = input[0].name` and `data_out = output[0].name`
(`Deeploy/Targets/Generic/Parsers.py:3312-3319`, class `RQSPerturbRademacherParser` at `3294`). In the update
graph those two names are **identical** (§1), so `data_in` and `data_out` resolve to the **same C symbol** →
the kernel adds the Rademacher perturbation directly onto the weight array in place. The `eps` / update
coefficient is folded into `mul` (per-channel int32) and, in the vendored runtime, scaled at run time
(`perturb_eps_override`).

---

## 3. The train ↔ update aliasing mechanism (same physical buffer, two networks)

Two networks are compiled **separately with different symbol prefixes** so they can be linked together:
- train graph → prefix `DeeployNetwork_` (emitted as `TrainingNetwork.c/.h`)
- update graph → prefix `DeeployOptNetwork_` / `DeeployOptimizerNetwork_` (emitted as `OptimizerNetwork.c/.h`)

Evidence for the two-network + prefix scheme (shipped standalone):
- `DeeployTest/generateOptimizerNetwork.py:10-12` — *"uses the prefix `DeeployOptNetwork_` (instead of the
  default `DeeployNetwork_`) so that it can be linked together with the training network without symbol
  conflicts."* → `name="DeeployOptimizerNetwork"` at `:71`.
- `DeeployTest/testRunner_tiled_siracusa_mezo.py:20` runs `--run_mode mezo_training`; the arg is declared in
  `DeeployTest/testMVP.py:221-222`.

### How the SAME weight buffer is shared — two variants

**Variant A — initializer layout (what the QMCUNetZO `.onnx` files actually encode).**
The weight is a *ConstantBuffer* in both networks, emitted as a `static PI_L2 … <weight_name>[…]` array
(§2b). Because both graphs use the **identical tensor name** (§1: all 44 update names ⊂ train names), and
Deeploy names its C globals after the tensor, the two translation units reference the **same global symbol**.
The train TU *defines* it; the update TU declares it `extern` and links to that one definition → both networks
mutate the **same L2 address**. No per-step copy is needed; the ZO update is literally in-place on the shared
constant array. (This is exactly what the vendored ZO harness header describes:
`TrainDeeploy/DeeployTest/Platforms/Siracusa/src/deeploymezotest.c:10-14` — *"the ZO update is performed
IN-PLACE on the shared weight constants by RunOptimizerNetwork (zo_update … ZERO inputs / ZERO outputs)"*,
and `:286-288` — *"Shares the training weight constants via name-matched buffer redirection"*.)

**Variant B — input layout (shipped BP/SGD harness, for contrast).**
When weights are graph *inputs* (`DeeployNetwork_inputs[]`), the shipped SGD harness does NOT alias — it copies
explicitly each optimizer step:
- `DeeployTest/Platforms/Siracusa/src/deeploytraintest.c:133-141` copies train weights+grads → optimizer inputs,
- `:147-150` `RunOptimizerNetwork`,
- `:159-161` copies `weight_updated` back into `DeeployNetwork_inputs[train_w_idx]`.
The vendored ZO/BP input-path replaces those copies with **pointer redirection** so no copy is needed:
`TrainDeeploy/DeeployTest/testUtils/codeGenerateTraining.py:579 _patch_shared_buffers`, doc `:598-600`:
```c
// Both malloc / arena-offset styles are rewritten to:
DeeployOptNetwork_input_N = (float32_t *)DeeployNetwork_input_M;   // direct pointer into Training's arena
```
driven by `build_shared_buffer_maps` (`:530`) which maps optimizer I/O index → training input index (input map
= reads, output map = in-place writes, `:611-614`).

**Net:** the shipped QMCUNetZO fixture uses **Variant A** (initializer + identical-name extern global). Variant B
is the alternative used when weights are inputs.

---

## 4. Shipped-initializer vs our-input approach comparison

| Aspect | Shipped QMCUNetZO (initializers) | Ours: Onnx4Deeploy feat/QZO (inputs) |
|---|---|---|
| Weight tensor kind | `graph.initializer` → `ConstantBuffer` | `graph.input` → `VariableBuffer(is_input=True)` (22 trainable weights) |
| (a) What makes it MUTABLE | ConstantBuffer is emitted as a **non-const writable `static PI_L2` array** (`AllocateTemplate.py:17-18,50`); the `const` template is commented out. Mutability is a *property of the emitted array*, no promotion step. | VariableBuffer inputs are `pi_l2_malloc`'d writable buffers by construction; the host seeds them and can read them back. Mutability is *inherent to inputs*. |
| (b) Train↔update ALIASING | Same **ONNX tensor name** in both graphs ⇒ same global C symbol ⇒ `extern` link to one definition ⇒ same L2 address. Update op has input==output name ⇒ in-place. | Same **tensor name** across the two graphs, aliased by name via the shared-buffer map / `_patch_shared_buffers` pointer redirection (`DeeployOptNetwork_input_N = DeeployNetwork_input_M`). |
| Update op in-place? | Yes — `RQSPerturbRademacher` input[0]==output[0]==weight name. | Yes if update op writes to the same-named input buffer; else host `memcpy` back (BP style, `deeploytraintest.c:159`). |
| Idiomatic to Deeploy? | Very — leans on Deeploy's existing ConstantBuffer emission (weights are always static arrays anyway). Zero harness plumbing for weights. | Also supported (BP training path is input-based) but needs explicit index maps / pointer patching or per-step copies. |
| Simplicity | Simpler at runtime: no `DeeployNetwork_inputs[]` weight slots, no copies, no index bookkeeping. Update graph has 0 inputs/0 outputs. | More moving parts: weight-input ordering, `TRAINING_NUM_WEIGHT_INPUTS`, `build_shared_buffer_maps`, pointer patch. |
| Robustness / risk | Robust: aliasing is by the linker (one symbol). Risk: relies on the `const` template staying disabled; and any pass that treats a ConstantBuffer as truly immutable (constant-folding, dedup, read-only placement, L3 read-only DMA) could break in-place mutation. Weights can't be re-seeded from host as easily (they're baked). | Robust: weights are first-class inputs the host can seed/inspect; no dependence on "constant means writable." Risk: correctness hinges on exact name/index alignment between the two graphs and on `_patch_shared_buffers` matching the emitted malloc/arena pattern; a codegen change to allocation style can silently break the redirect. Also more host-side bookkeeping and (BP style) potential double-buffering copies. |

**Which is better?** For *this* ZO use-case the shipped **initializer** design is the more idiomatic and the
leanest at runtime — the update graph is inputs=0/outputs=0 and mutation is pure in-place on symbols the linker
already unifies. Our **input** design is more explicit and host-visible (easy to seed/read/checkpoint weights,
no reliance on "ConstantBuffer is secretly writable"), at the cost of name/index-alignment plumbing and the
`_patch_shared_buffers` regex-matching fragility. Our approach *avoids* the risk that a future
constant-immutability optimization (const placement / read-only L3 / constant dedup) silently breaks the
in-place weight update; it *introduces* the risk that the two graphs' input orderings / arena-allocation
patterns must stay in lockstep for the pointer aliasing to be correct.

---

## 5. Answer to the puzzle (TL;DR restated)

1. **Two graphs.** `QMCUNetZO/network.onnx` = ZO **train** (inputs `input`,`label`; 336 initializers; 84
   RQSPerturbRademacher + Conv/RequantShift/…/SoftmaxCrossEntropyLoss). `QMCUNetZO/network_zo_update.onnx` = ZO
   **update** (0 inputs, 0 outputs, 44 initializers, 44 RQSPerturbRademacher). Trainable weights are
   **initializers** in *both*.
2. **Initializers are not constant.** Deeploy imports them as `ConstantBuffer` but emits them as
   **writable `static PI_L2 … w[…] = {…}` arrays** (`AllocateTemplate.py:17-18,50`; the `const` template at line
   23 is disabled). The ZO update op has **input name == output name == weight name**, so the kernel
   (`RQSPerturbRademacherTemplate.py:46-53`) reads and writes the same buffer — genuine in-place mutation.
3. **Train↔update alias the same buffer** because both graphs use the **identical ONNX tensor name**, which
   Deeploy turns into the **same global C symbol**; the update network (prefix `DeeployOptNetwork_`,
   `generateOptimizerNetwork.py:10-12`) is linked against the train network (prefix `DeeployNetwork_`) and the
   weight arrays resolve (via `extern`) to one L2 address — *"the ZO update is performed IN-PLACE on the shared
   weight constants"* (`deeploymezotest.c:10-14`). No per-step weight copy.
