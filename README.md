# DaKanji: Single Character Recognition

Recognize a single handwritten or printed Japanese character (**Kanji, Hiragana, or
Katakana**) from a drawing. The pipeline is language-agnostic, so with your own data it
works for other scripts (e.g. Chinese) too.

<img src="https://user-images.githubusercontent.com/51273483/233300113-502930e9-dcac-4f54-b522-9e186906da14.gif" style="display:block;margin-left:auto;margin-right:auto;" width="25%"/>

A small, fast CNN (EfficientNet-Lite) that takes a **grayscale image of any size** and
returns class probabilities aligned with [`labels.txt`](labels.txt). Resize + normalize
are baked **into the model**, so it deploys to ONNX / ExecuTorch with zero preprocessing
code in your app.

## Highlights

- 🔥 **PyTorch** + **Marimo** notebooks, managed with **uv** (Ruff + mypy, fully typed).
- 🧠 **Plug-and-play models**: EfficientNet-Lite (timm), MobileNetV3, a tiny baseline.
- 🖼️ **Preprocessing inside the model**: feed a raw image of any size, get probabilities.
- 🧩 **Generic data loader**: just point it at `root/<class>/*.png` + a label list.
- 📉 **Two-stage int8 quantization**: train fp32, then PT2E QAT → an XNNPACK `.pte` that runs on device.
- 📦 **Export to ONNX & ExecuTorch** with dynamic input shapes (one artifact, any size).

## Quick start

```bash
# Install uv (https://docs.astral.sh/uv), then:
uv sync --extra dev                                # core + dev/test deps (runs the full suite)
uv run pytest -q                                   # verify on synthetic data (no dataset needed)

# Train a quick model on random data, or open the interactive notebook:
uv run python scripts/train.py --synthetic --epochs 3
uv run marimo edit notebooks/train.py
```

## Workflow (real data)

Lay your images out as `<root>/<class_index>/*.{jpg,png}` and point the data config at
them — [configs/data/kanji.toml](configs/data/kanji.toml): `root` is the train set,
`val_root` the held-out set. Then run the pipeline:

```bash
# 0. (optional) watch metrics live — separate terminal. Port 5001 dodges macOS AirPlay on 5000.
uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db --port 5001   # http://127.0.0.1:5001

# 1. Train the fp32 model (90/10 split; writes outputs/runs/best.pt, logs to MLflow)
uv run python scripts/train.py --config configs/runs/kanji.toml --epochs 30

# 2. Evaluate fp32 on the held-out set (top-1/5/10 + outputs/eval/predictions.png grid)
uv run python scripts/evaluate.py --config configs/runs/kanji.toml

# 3. Quantize + QAT fine-tune -> int8 XNNPACK .pte (outputs/exports/model_xnnpack.pte)
uv run python scripts/quantize.py --config configs/runs/kanji.toml --qat-epochs 8 --lr 1e-5

# 4. Evaluate the int8 .pte on the same held-out set (compare to step 2)
uv run python scripts/evaluate.py --config configs/runs/kanji.toml --pte outputs/exports/model_xnnpack.pte

# 5. Export the fp32 model to ONNX + ExecuTorch .pte (with PyTorch parity check)
uv run python scripts/export.py --from outputs/runs/best.pt
```

The fp32 `best.pt` stays the source of truth: re-quantize or re-export from it any time.
(Step 4 runs the `.pte` per image through the ExecuTorch runtime, so it's slower than the
fp32 eval — point `--root` at a subset for a quick check.)

### Advanced

```bash
# Sweep backbone x input size (fp32) to pick an architecture; results -> CSV + MLflow
uv run python scripts/grid_search.py --config configs/runs/grid.toml

# Or experiment interactively (widgets for backbone / size / lr; logs to MLflow)
uv run marimo edit notebooks/train.py
```

> This repo trains on a **pre-rendered** image dataset; it doesn't generate the dataset for
> you. Training data for the Japanese model came from the
> [ETL Character Database](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database) and
> [KanjiVG](https://kanjivg.tagaini.net/).

## How it works

1. A **run config** (`configs/runs/*.toml`) holds the training recipe inline (model,
   optimizer, augmentation) and references reusable `data` / `log` fragments by name.
2. **Training** runs a custom PyTorch loop (AMP, schedules, MLflow logging, checkpoints),
   producing an fp32 checkpoint as the source of truth.
3. **Quantization** (optional, stage 2) fine-tunes that checkpoint with **PT2E QAT** and
   lowers it to an **int8 XNNPACK `.pte`** for on-device CPU/ARM inference.
4. **Export** turns a checkpoint into an ONNX or ExecuTorch artifact that accepts a raw
   grayscale image of any size and outputs character probabilities.

**Inference contract.** Input: grayscale `(B, 1, H, W)`, any size, any range. Output:
probabilities `(B, num_classes)` aligned with `labels.txt`.

## Documentation

The full guide (installation, data loaders, model registry, training, experimentation,
quantization, export & deployment, and code quality) lives in **[TECHNICAL.md](TECHNICAL.md)**.
See [CHANGELOG.md](CHANGELOG.md) for what changed in v2.

## Apps using this model

| name | Android | iOS | Linux | macOS | Windows | Web |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DaKanji | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| Kanji Graph | ✅ |  |  |  |  |  |

Using it in your own software is very welcome. A credit like
`Character recognition powered by machine learning from CaptainDario (DaAppLab)` is
appreciated, and feel free to open an issue so it can be added to the list above.

## Credits

Training data kindly provided by the
[ETL Character Database](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database) and
[the KanjiVG dataset](https://kanjivg.tagaini.net/). Architecture based on
[EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946).

## License

[MIT](LICENSE) © CaptainDario
