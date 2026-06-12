# DaKanji — Single Character Recognition (PyTorch + Marimo)

Recognize a single handwritten/printed character (Kanji, Hiragana, Katakana — and,
because the pipeline is language agnostic, any script you have data for).

This is a ground-up rewrite of the original TensorFlow/Keras + Jupyter project:

| | before | now |
|---|---|---|
| framework | TensorFlow / Keras | **PyTorch** |
| notebooks | Jupyter `.ipynb` | **Marimo** reactive `.py` |
| env / tooling | `pip` + `requirements.txt` | **uv** + `pyproject.toml`, **Ruff**, **mypy** |
| preprocessing | in app / in graph | **inside the model** (deploy with zero in-app preprocessing) |
| augmentation | keras_cv + custom layers | **torchvision.transforms.v2** (MixUp/CutMix/shear/sharpness/cutout) |
| logging | Keras callbacks | manual **MLflow** |
| export | TFLite | **ONNX + ExecuTorch** (dynamic input shapes) |
| optimization | TF Model Optimization | **PT2E QAT** (int8, conv+linear) → XNNPACK `.pte`, runs on device |

The key deployment property is preserved and made explicit: **resize + normalize live
inside the model**, so the exported ONNX/ExecuTorch graph accepts a *raw grayscale
image of any size* and needs no preprocessing code on device.

---

## Quick start

```bash
# 1. Install uv (https://docs.astral.sh/uv/) if you don't have it, then:
uv sync                       # core deps (PyTorch, torchvision, marimo, ...)
uv sync --extra onnx --extra dev   # + ONNX export and dev tooling

# 2. Verify everything works on random synthetic data (no dataset needed):
uv run pytest -q
uv run python scripts/train.py --synthetic --classes 20 --epochs 3

# 3. Open the interactive training notebook:
uv run marimo edit notebooks/train.py
```

Optional extras: `--extra tracking` (MLflow), `--extra optimize` (TorchAO),
`--extra executorch` (on-device export).

---

## Project structure

```text
configs/               # TOML config fragments, split by type, composed by a run config
  data/ model/ optim/ augment/ log/   # one fragment per type (e.g. data/kanji.toml)
  runs/                # run configs that reference one fragment of each (+ [grid])
outputs/               # all generated artifacts (checkpoints, exports, MLflow) — gitignored
src/char_recognition/
  paths.py             # central paths: configs/fonts/labels + OUTPUTS_DIR (change in one place)
  config/              # one module per config: data / model / augment / optim / log + loader
  data/
    labels.py          # load class labels (one-per-line, or legacy one-char-per-line)
    dataset.py         # CharFolderDataset — generic root/<class>/*.png loader -> (C,H,W)
    synthetic.py       # RandomCharDataset — learnable random data for tests
    datamodule.py      # split + DataLoaders (caching/prefetch equivalents)
    augment.py         # transforms.v2 pipeline + MixUp/CutMix collate
  models/
    preprocessing.py        # Preprocess: in-model resize + normalize (exported with the graph)
    timm_backbones.py       # EfficientNet-Lite0 from timm
    torchvision_backbones.py# MobileNetV3 small/large adapted to grayscale
    tiny_cnn.py             # fast baseline
    registry.py             # plug-and-play backbone registry (select by name)
    recognizer.py           # CharRecognizer = Preprocess + backbone; ProbabilityModel
  engine/
    trainer.py         # custom AMP train/val loop (replaces model.fit)
    optim.py           # optimizer / scheduler / loss factories
    checkpoint.py      # torch.save checkpointing + rebuild metadata
    logger.py          # manual MLflow logging
  optimize/
    pt2e.py            # PT2E int8 quantization: graph capture + prepare/convert (QAT or PTQ)
    benchmark.py       # latency + serialized-size benchmark
  export/
    onnx.py            # ONNX export (dynamic batch/H/W)
    executorch.py      # ExecuTorch .pte export, fp32 (dynamic H/W)
    xnnpack.py         # XNNPACK .pte export, int8 (lowers a PT2E-converted graph)
    loading.py         # rebuild a model from a checkpoint; example inputs; softmax wrapper
  engine/runner.py     # wire a Config into a training run (shared by CLI/grid/notebook)
notebooks/
  train.py             # interactive training notebook (Marimo): config + tweak widgets
  export_onnx.py       # export to ONNX + parity check
  export_executorch.py # export to ExecuTorch
scripts/
  train.py             # stage 1: headless fp32 training CLI
  quantize.py          # stage 2: PT2E QAT fine-tune of a checkpoint -> XNNPACK .pte
  grid_search.py       # sweep model x input size (fp32, config-driven)
tests/                 # end-to-end checks on random data (models/data/engine/export/backend)
labels.txt             # ordered class labels for the Japanese model
fonts/                 # CJK font for rendering predictions in matplotlib
```

---

## Data loaders

The loader contract is intentionally minimal — **labels + a folder, nothing more**:

```text
root/
  0/  img0.png  img1.png ...     # class index 0  -> labels[0]
  1/  ...                        # class index 1  -> labels[1]
  ...
```

```python
from char_recognition.data import CharFolderDataset, load_labels

labels = load_labels("labels.txt")                 # list[str], index == class id
dataset = CharFolderDataset("/data/kanji", image_size=(64, 64), in_channels=1)
image, target = dataset[0]                          # image: (C, H, W) float in [0, 255]
```

Class folders named with integers are sorted numerically (the ETL/DaKanji layout);
otherwise lexicographically. To reuse the project for **Chinese** or any other script,
point it at a new folder and pass a matching `labels.txt`. Nothing else changes.

`RandomCharDataset` produces a *learnable* synthetic dataset (a fixed random prototype
per class + noise) so the entire pipeline can be verified without any real data.

`tf.data` knobs map to: `prefetch(AUTOTUNE)` → `num_workers` + `prefetch_factor`,
warm workers → `persistent_workers`, pinned memory → `pin_memory`. Note there is no
decoded-tensor cache (`tf.data.cache()`); images are re-decoded each epoch, which the
OS page cache mitigates but doesn't eliminate.

---

## Models (plug-and-play)

Backbones are selected by name from a registry; swap them without touching training code.

```python
from char_recognition.models import available_backbones, CharRecognizer
available_backbones()
# ['efficientnet_lite_b0', 'mobilenetv3_large', 'mobilenetv3_small', 'tiny_cnn']

model = CharRecognizer(num_classes=3036, backbone="efficientnet_lite_b0", image_size=(64, 64))
```

| backbone | params* | notes |
|---|---:|---|
| `efficientnet_lite_b0` | ~3.4M | the **original** architecture, via timm's `tf_efficientnet_lite0` |
| `mobilenetv3_small` | ~1.6M | modern, smallest — great on-device target |
| `mobilenetv3_large` | ~4.3M | modern, strongest of the set |
| `tiny_cnn` | ~0.3M | fast baseline |

\* at 50 classes; the classifier grows with the class count.

> torchvision has no EfficientNet-*Lite* variant (only standard EfficientNet / V2), so
> `efficientnet_lite_b0` uses [timm](https://github.com/huggingface/pytorch-image-models)'s
> `tf_efficientnet_lite0` — the same architecture (and TF-style padding) as the original.

Register your own with `register_backbone("name", builder)` where `builder` has the
signature `(num_classes, in_channels=1, **kwargs) -> nn.Module`.

### Configurable input size

Input size is a first-class config value, so you can experiment with 48 / 64 / 96 / 128:

```python
CharRecognizer(num_classes, backbone="efficientnet_lite_b0", image_size=(128, 128))
```

The model's `Preprocess` resizes any raw input to this size internally, so training,
evaluation and the exported artifact all agree.

---

## Training

Hyperparameters live in version-controlled TOML fragments under `configs/`, split by
type and composed by a **run config** that references one fragment of each. Configs are
logged to MLflow so every run is reproducible:

```toml
# configs/runs/kanji.toml      (a run = one fragment per type)
device = "auto"
data    = "kanji"              # -> configs/data/kanji.toml (root, labels, image_size, ...)
model   = "efficientnet_lite_b0"
optim   = "default"
augment = "default"
log     = "default"
```

```toml
# configs/data/kanji.toml      (the referenced dataset fragment)
root = "/data/kanji"           # dataset folder: root/<class_index>/*.png
labels_file = "labels.txt"
image_size = [64, 64]
```

A fragment can be reused by many runs (e.g. one `data/kanji.toml`, several models).

### Interactive (Marimo)

```bash
uv run marimo edit notebooks/train.py
```

Pick a base config from a dropdown, then tweak backbone / input size / epochs / lr /
MixUp with widgets for quick experiments. The **effective** config (base + tweaks) is
logged to MLflow, so interactive runs stay tracked. Heavy training is gated behind a
**▶ Train** button; the notebook ends with the training curves and a **10×10 grid of
predictions** rendered straight into the output.

### Headless (CLI)

```bash
uv run python scripts/train.py \
    --data-root /data/kanji --labels labels.txt \
    --backbone efficientnet_lite_b0 --image-size 64 \
    --batch-size 256 --epochs 50 --lr 1e-3
```

### What replaces `model.fit()`

A custom loop in `engine/trainer.py` implements, explicitly:

- **Mixed precision** via `torch.amp.autocast`. On CUDA this is fp16 + `GradScaler`
  (the direct equivalent of Keras `mixed_float16`); on MPS/CPU it is bf16.
- **Scheduling** — `cosine` (default, with warmup) or `step_decay` (reproduces the
  original 4 %-every-3-epochs rule).
- **Checkpointing** — manual `torch.save` of model/optimizer/scheduler + metadata,
  tracking `best.pt` and `last.pt` (replaces `ModelCheckpoint`).
- **Logging** — manual MLflow logging of params, metrics and figures (replaces the
  Keras TensorBoard callback). Enabled via config; skipped with a warning if MLflow
  isn't installed (`uv sync --extra tracking`).

---

## Augmentation

`augment.py` translates the keras_cv pipeline to `torchvision.transforms.v2`:

- per-sample: **RandomShear**, **RandomSharpness**, **RandomCutout** (random erasing),
- batch-level: **MixUp** and **CutMix** (native v2 support) applied in the collate fn,
  producing soft labels that the cross-entropy loss consumes directly.

Augmentation is applied only to the training split; validation always sees clean images.

---

## Experimentation

Quick, interactive experiments go in the **training notebook** (tweak the widgets,
read MLflow). Systematic sweeps over **model × input size** use the **grid-search
script**, driven by a config:

```bash
uv run python scripts/grid_search.py --config configs/runs/grid.toml
```

Each `(backbone, image_size)` is a separate fp32 training run; the grid reports accuracy,
latency and serialized size, logs each point to MLflow, and writes a CSV / ranked table.
Edit the `[grid]` section to choose what to sweep:

```toml
[grid]
backbones = ["efficientnet_lite_b0", "mobilenetv3_small"]
image_sizes = [64, 128]
```

Quantization is *not* part of the sweep — train fp32 first, then quantize the winner
(see below). Keeping the two stages separate means one fp32 source of truth per
architecture and a cheap, repeatable quantization pass on top.

---

## Optimization — two-stage int8 quantization (PT2E QAT → XNNPACK)

Quantization is a **second stage**, deliberately decoupled from training:

1. **Stage 1 — `scripts/train.py`** trains the fp32 model. The fp32 checkpoint is the
   source of truth; you keep it, re-quantize from it, and never lose precision to a
   one-way conversion.
2. **Stage 2 — `scripts/quantize.py`** takes that checkpoint, runs a short
   **PT2E Quantization-Aware fine-tune** (int8), converts to a real int8 graph, and
   lowers it to an **XNNPACK ExecuTorch `.pte`** that runs on CPU/ARM.

```bash
# stage 1 (elsewhere): uv run python scripts/train.py --config configs/runs/kanji.toml
uv sync --extra optimize --extra executorch
uv run python scripts/quantize.py --config configs/runs/kanji.toml --qat-epochs 8 --lr 1e-5
```

**Why PT2E and not the module-swap quantizers?** PT2E is graph-based: it captures the
model with `torch.export` and quantizes **Conv2d *and* Linear** (with conv-BN fusion).
The module-swap path only touches Linear, which barely moves the needle on these
conv-heavy backbones. **Why QAT and not plain PTQ?** Fine-tuning with fake-quant in the
loop recovers the accuracy int8 PTQ tends to drop; `quantize.py` keeps it short and at a
low LR because it starts from converged fp32 weights.

The model is wrapped in `ProbabilityModel` (softmax) **before** capture, so the lowered
`.pte` emits probabilities directly — a converted graph can't be re-wrapped afterwards.
Capture and re-export share one dynamic-shape spec (`optimize.pt2e.dynamic_input_shapes`):
dynamic batch + H/W, captured with a batch-`CAPTURE_BATCH` example (a batch-1 example
would 0/1-specialize the batch dim and break it). The deployed graph still runs at batch 1
and any H/W. `tests/test_backend_support.py` lowers an int8 model and runs it in the
XNNPACK runtime at two input sizes to guard this end to end.

> Trade-offs: int8 has a small accuracy cost vs fp32 (the QAT fine-tune narrows it);
> the export pins the deployment backend to XNNPACK (CPU/ARM). CoreML/ANE int8 is out of
> scope — coremltools currently can't lower the int8 cast (fp16 CoreML works, int8 does not).

---

## Export & deployment

Because preprocessing is in the model, **one artifact serves every runtime input size**.
There are three export targets:

- **ONNX** — fp32, dynamic batch + H/W (`export_onnx`, notebook `export_onnx.py`).
- **ExecuTorch fp32** — `.pte`, dynamic H/W (`export_executorch`, notebook `export_executorch.py`).
- **ExecuTorch int8 / XNNPACK** — `.pte` produced by the **stage-2 `scripts/quantize.py`**
  pipeline above (`export_xnnpack` lowers the PT2E-converted graph). This is the on-device
  deployment artifact.

The fp32 exports run from dedicated Marimo notebooks (point them at a checkpoint):

```bash
# ONNX (dynamic batch + height + width); also checks parity vs PyTorch
uv run marimo edit notebooks/export_onnx.py

# ExecuTorch fp32 (.pte, dynamic height/width)
uv sync --extra executorch
uv run marimo edit notebooks/export_executorch.py
```

> All three exports are covered by the test suite — the ExecuTorch/XNNPACK tests load the
> `.pte` in the runtime and run it at multiple input sizes. The `executorch` package is
> optional and version-sensitive (`uv sync --extra executorch`); those tests skip when it's absent.

Programmatically, `export_onnx` / `export_executorch` from `char_recognition.export`
do the same:

```python
from char_recognition.export import export_onnx, load_recognizer

model = load_recognizer("runs/kanji/best.pt")
export_onnx(model, "exports/model.onnx", image_size=model.image_size, dynamic=True)
```

### Inference contract

- **Input:** a grayscale image `(B, 1, H, W)` of any size, any value range. The model
  resizes and normalizes it internally.
- **Output:** class probabilities `(B, num_classes)` aligned with `labels.txt`.

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession("exports/model.onnx")
raw = np.random.randint(0, 256, (1, 1, 300, 90)).astype("float32")  # any H, W
probs = sess.run(None, {"image": raw})[0]                            # (1, num_classes)
```

---

## Results & accuracy

`efficientnet_lite_b0` is the same architecture as the original Keras model (timm's
`tf_efficientnet_lite0`), so trained on the same ETL/KanjiVG data with the recipe above
it should reach accuracy in the same ballpark as the original (~0.95 val accuracy in the
legacy project). This is an expectation, not a guarantee — weights aren't ported, and
the in-model normalization differs (per-image max vs the original's per-batch max).
Reproducing the exact number needs the original dataset, which isn't bundled; the
included tests instead prove the full loop *learns* on synthetic data. Use
`scripts/grid_search.py` to compare backbones/input sizes on your data.

---

## Code quality

```bash
uv run ruff check src tests scripts     # lint (pycodestyle, pyflakes, isort, bugbear, annotations, ...)
uv run ruff format src tests scripts    # format
uv run mypy                             # type-check the package
uv run pytest -q                        # end-to-end tests on random data
```

PEP-8, strict type hints, explicit device placement, one purpose per module.

> Tip: point your editor's Python interpreter at `.venv/bin/python` so it resolves the
> installed packages (uv creates `.venv` on `uv sync`).

---

## Credits

Training data kindly provided by the [ETL Character Database](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database)
and [the KanjiVG dataset](https://kanjivg.tagaini.net/).

If you use this model, credit is appreciated:
`Character recognition powered by machine learning from CaptainDario (DaAppLab)`.

Papers:

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Recognizing Handwritten Japanese Characters Using Deep Convolutional Neural Networks](http://cs231n.stanford.edu/reports/2016/pdfs/262_Report.pdf)
