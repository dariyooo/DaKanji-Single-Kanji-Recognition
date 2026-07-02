# DaKanji: Single Character Recognition

A small, fast CNN that takes a **(grayscale) image of any size** and returns character probabilities.
Recognize a single handwritten or printed Japanese character (**Kanji, Hiragana, or
Katakana**) from a drawing. The pipeline is language-agnostic, so with your own data it
works for other scripts (e.g. Chinese) too.

<img src="https://user-images.githubusercontent.com/51273483/233300113-502930e9-dcac-4f54-b522-9e186906da14.gif" style="display:block;margin-left:auto;margin-right:auto;" width="25%"/>

This is the complete v2 rewrite of this project where I applied all my learnings from the last years.

* Data agnostic: model for Japanese / Chinese / Korean / ...
* Preprocessing in the model: No more tedious image handling in app code, everything as high-performance Tensor operations
* Latest frameworks: Uses Torch + MLflow + Marimo for dev and ONNX / ExecuTorch for Deployment
* Real model comparison: Benchmarked Custom CNN / MobileNet / EffNet Lite on accuracy and latency
* Proper optimization: Latest quantization and tuned embedding size
* Result: A tiny custom CNN (<3MB) with sub millisecond latency (~0.2ms on m5 max)

| backend | precision | latency(ms) | size(MB) |
|---|---|---|---|
| ExecuTorch portable (CPU) | fp32 | 36.60 | 8.20 |
| XNNPACK (CPU) | int8 | 0.15 | 2.16 |
| CoreML (ANE/GPU/CPU) | fp16 | 0.15 | 4.20 |

[Full results](#full-results)

## Quick start

Put your labled images in a directory `<root>/<class_index>/*.{jpg,png}` and point the data config at
them. [configs/data/kanji.toml](configs/data/kanji.toml): `root` is the train set,
`val_root` the held-out set. Then run the pipeline:

```bash
# 0. (optional) watch metrics live — separate terminal.
uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db --port 5001   # http://127.0.0.1:5001

# 1. Train the fp32 model (90/10 split; writes outputs/runs/best.pt, logs to MLflow)
uv run python scripts/train.py --config configs/runs/kanji_efficientnet_lite_b0.toml --epochs 30

# 2. Evaluate fp32 on the held-out set (top-1/5/10 + outputs/eval/predictions.png grid)
uv run python scripts/evaluate.py --config configs/runs/kanji_efficientnet_lite_b0.toml --from outputs/runs/best.pt

# 3. Quantize + QAT fine-tune -> int8 XNNPACK .pte (outputs/exports/model_xnnpack.pte)
uv run python scripts/quantize.py --config configs/runs/kanji_efficientnet_lite_b0.toml --from outputs/runs/best.pt --qat-epochs 8 --lr 1e-5

# 4. Evaluate the int8 .pte on the same held-out set (compare to step 2)
uv run python scripts/evaluate.py --config configs/runs/kanji_efficientnet_lite_b0.toml --pte outputs/exports/model_xnnpack.pte

# 5. Export the fp32 model to ONNX + ExecuTorch .pte (with PyTorch parity check)
uv run python scripts/export.py --from outputs/runs/best.pt
```

The fp32 `best.pt` stays the source of truth: re-quantize or re-export from it any time.
(Step 4 runs the `.pte` per image through the ExecuTorch runtime, so it's slower than the
fp32 eval — point `--root` at a subset for a quick check.)

### Advanced

```bash
# Grid search backbone x input size (fp32) to pick an architecture; results -> CSV + MLflow
uv run python scripts/architecture_search.py --config configs/runs/grid.toml

# Or experiment interactively (widgets for backbone / size / lr; logs to MLflow)
uv run marimo edit notebooks/train.py

# Apple CoreML .pte (ANE/GPU/CPU). Self-contained: uv pins an isolated torch 2.7 toolchain
# from the script header, because the CoreML delegate miscompiles on the main torch 2.12 stack.
uv run scripts/export_coreml.py --from outputs/runs/best.pt
```

### Packaging for the DaKanji app

[`scripts/prepare_release_assets.py`](scripts/prepare_release_assets.py) bundles every exported
artifact (the per-backend `.pte`, ONNX, and labels) into the release assets the app expects, and
throws if any are missing:

```bash
uv run python scripts/prepare_release_assets.py   # -> outputs/release/
```

## How it works

1. A **run config** (`configs/runs/*.toml`) holds the training config inline (model, optimizer, augmentation, logging) and references one reusable `data` fragment by name.
2. **Training** runs a custom PyTorch loop (AMP, schedules, MLflow logging, checkpoints), producing an fp32 checkpoint as the source of truth.
3. **Quantization** (optional, stage 2) fine-tunes that checkpoint with **PT2E QAT** and lowers it to an **int8 XNNPACK `.pte`** for on-device CPU/ARM inference.
4. **Export** turns a checkpoint into an ONNX or ExecuTorch artifact that accepts a raw grayscale image of any size and outputs character probabilities.

**Inference.**
Input: an image `(B, C, H, W)` with `C = 1` (grayscale, preferred) or `C = 3` (RGB, may have slightly worse results), any size, any range. The model resizes and reduces colour to one channel internally.
Output: probabilities `(B, num_classes)` aligned with `labels.txt`.

## Apps that use this model

| name | android | iOS | Linux | MacOS | Windows | Web |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DaKanji (v4+) | ✅ | ✅ | ✅ | ✅ | ✅ |   |

## Full Results

Results of initial trainig

| backbone (fp32) | size | acc | latency(ms) | size(MB) |
|---|---|---|---|---|
| efficientnet_lite_b4 | 64 | 0.997 | 402.15 | 80.88 |
| **small_cnn** | **64** | **0.995** | **0.74** | **11.40** |
| mobilenetv3_large | 64 | 0.995 | 59.50 | 50.35 |
| efficientnet_lite_b0 | 64 | 0.992 | 142.36 | 47.09 |
| mobilenetv3_small | 64 | 0.990 | 52.87 | 32.88 |

*(latency = mean of 300 runs)*

Results of embedding tuning

| head_rank | acc | latency(ms) | size(MB) | params(M) |
|---|---|---|---|---|
| 256 | 0.995 | 0.74 | 11.67 | 2.91 |
| 192 | 0.995 | 0.90 | 9.94 | 2.48 |
| **128** | **0.995** | **0.85** | **8.20** | **2.04** |
| 96 | 0.994 | 0.72 | 7.34 | 1.83 |
| 64 | 0.993 | 0.72 | 6.47 | 1.61 |

Results of quantization (top-1/5/10 on a 20k val_root sample)

| precision | top-1 | top-5 | top-10 | latency(ms) | size(MB) | params(M) |
|---|---|---|---|---|---|---|
| fp32 | 0.9973 | 1.0000 | 1.0000 | 0.71 | 8.20 | 2.04 |
| fp16 | 0.9972 | 1.0000 | 1.0000 | ~0.71 | 4.10 | 2.04 |
| **int8 (XNNPACK)** | **0.9948** | **1.0000** | **1.0000** | **0.15** | **2.16** | **2.04** |
| int8 (ONNX) | 0.9965 | 1.0000 | 1.0000 | 0.15 | 2.23 | 2.04 |

Backend comparison

| backend | precision | latency(ms) | size(MB) |
|---|---|---|---|
| ExecuTorch portable (CPU) | fp32 | 36.60 | 8.20 |
| XNNPACK (CPU) | int8 | 0.15 | 2.16 |
| ONNX Runtime (CPU) | int8 | 0.15 | 2.23 |
| CoreML (ANE/GPU/CPU) | fp16 | 0.15 | 4.20 |

## Usage in your software

I put lots of effort and time into developing this model and hope that it can be used in many apps.
If you decide to use this machine learning model please give me credit like:
`Character recognition powered by machine learning from Dariyooo (DaAppLab)`
It would also be nice if you open an issue and tell me that you are using this model.
Than I would add your software to the [apps section](#apps-which-use-this-model)

## Credits

Training data kindly provided by the
[ETL Character Database](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database) and
[the KanjiVG dataset](https://kanjivg.tagaini.net/). Architecture based on
[EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946).

## License

[MIT](LICENSE) © CaptainDario
