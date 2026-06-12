# DaKanji — Single Character Recognition

Recognize a single handwritten or printed Japanese character — **Kanji, Hiragana, or
Katakana** — from a drawing. The pipeline is language-agnostic, so with your own data it
works for other scripts (e.g. Chinese) too.

<img src="https://user-images.githubusercontent.com/51273483/233300113-502930e9-dcac-4f54-b522-9e186906da14.gif" style="display:block;margin-left:auto;margin-right:auto;" width="25%"/>

A small, fast CNN (EfficientNet-Lite) that takes a **grayscale image of any size** and
returns class probabilities aligned with [`labels.txt`](labels.txt). Resize + normalize
are baked **into the model**, so it deploys to ONNX / ExecuTorch with zero preprocessing
code in your app.

## Highlights

- 🔥 **PyTorch** + **Marimo** notebooks, managed with **uv** (Ruff + mypy, fully typed).
- 🧠 **Plug-and-play models** — EfficientNet-Lite (timm), MobileNetV3, a tiny baseline.
- 🖼️ **Preprocessing inside the model** — feed a raw image of any size, get probabilities.
- 🧩 **Generic data loader** — just point it at `root/<class>/*.png` + a label list.
- 📉 **Two-stage int8 quantization** — train fp32, then PT2E QAT → an XNNPACK `.pte` that runs on device.
- 📦 **Export to ONNX & ExecuTorch** with dynamic input shapes (one artifact, any size).

## Quick start

```bash
# Install uv (https://docs.astral.sh/uv), then:
uv sync                                            # core dependencies
uv run pytest -q                                   # verify on synthetic data (no dataset needed)

# Train a quick model on random data, or open the interactive notebook:
uv run python scripts/train.py --synthetic --epochs 3
uv run marimo edit notebooks/train.py
```

**Train on real data.** Arrange images as `root/<class_index>/*.png`, point a config at
the folder, and run:

```bash
uv run python scripts/train.py --config configs/runs/kanji.toml
```

> The model trains on a **pre-rendered** image dataset (`root/<class>/*.png`) — this repo
> doesn't generate the dataset for you. Training data for the Japanese model came from the
> [ETL Character Database](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database)
> and [KanjiVG](https://kanjivg.tagaini.net/).

## How it works

1. A **config** (`configs/runs/*.toml`) selects the dataset, backbone, input size and
   training recipe by composing small typed fragments.
2. **Training** runs a custom PyTorch loop (AMP, schedules, MLflow logging, checkpoints),
   producing an fp32 checkpoint as the source of truth.
3. **Quantization** (optional, stage 2) fine-tunes that checkpoint with **PT2E QAT** and
   lowers it to an **int8 XNNPACK `.pte`** for on-device CPU/ARM inference.
4. **Export** turns a checkpoint into an ONNX or ExecuTorch artifact that accepts a raw
   grayscale image of any size and outputs character probabilities.

**Inference contract** — input: grayscale `(B, 1, H, W)`, any size, any range. Output:
probabilities `(B, num_classes)` aligned with `labels.txt`.

## Documentation

The full guide — installation, data loaders, model registry, training, experimentation,
quantization, export & deployment, and code quality — lives in **[TECHNICAL.md](TECHNICAL.md)**.
See [CHANGELOG.md](CHANGELOG.md) for what changed in v2.

## Apps using this model

| name | Android | iOS | Linux | macOS | Windows | Web |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DaKanji | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| Kanji Graph | ✅ |  |  |  |  |  |

Using it in your own software is very welcome — a credit like
`Character recognition powered by machine learning from CaptainDario (DaAppLab)` is
appreciated, and feel free to open an issue so it can be added to the list above.

## Credits

Training data kindly provided by the
[ETL Character Database](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database) and
[the KanjiVG dataset](https://kanjivg.tagaini.net/). Architecture based on
[EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946).

## License

[MIT](LICENSE) © CaptainDario
