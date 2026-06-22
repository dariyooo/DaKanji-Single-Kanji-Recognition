# DaKanji: Single Character Recognition

Recognize a single handwritten or printed Japanese character (**Kanji, Hiragana, or
Katakana**) from a drawing. The pipeline is language-agnostic, so with your own data it
works for other scripts (e.g. Chinese) too.

<img src="https://user-images.githubusercontent.com/51273483/233300113-502930e9-dcac-4f54-b522-9e186906da14.gif" style="display:block;margin-left:auto;margin-right:auto;" width="25%"/>

A small, fast CNN that takes a **grayscale image of any size** and
returns class probabilities aligned with [`labels.txt`](labels.txt). Resize + normalize
are baked **into the model**, so it deploys to ONNX / ExecuTorch with zero preprocessing
code in your app (TF lite models are now deprecated, revert to a previous release for them).

## Quick start

Lay your images out as `<root>/<class_index>/*.{jpg,png}` and point the data config at
them — [configs/data/kanji.toml](configs/data/kanji.toml): `root` is the train set,
`val_root` the held-out set. Then run the pipeline:

```bash
# 0. (optional) watch metrics live — separate terminal. Port 5001 dodges macOS AirPlay on 5000.
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
# Sweep backbone x input size (fp32) to pick an architecture; results -> CSV + MLflow
uv run python scripts/grid_search.py --config configs/runs/grid.toml

# Or experiment interactively (widgets for backbone / size / lr; logs to MLflow)
uv run marimo edit notebooks/train.py

# Apple CoreML .pte (ANE/GPU/CPU). Self-contained: uv pins an isolated torch 2.7 toolchain
# from the script header, because the CoreML delegate miscompiles on the main torch 2.12 stack.
uv run scripts/export_coreml.py --from outputs/runs/best.pt
```

## How it works

1. A **run config** (`configs/runs/*.toml`) holds the training recipe inline (model,
   optimizer, augmentation, logging) and references one reusable `data` fragment by name.
2. **Training** runs a custom PyTorch loop (AMP, schedules, MLflow logging, checkpoints),
   producing an fp32 checkpoint as the source of truth.
3. **Quantization** (optional, stage 2) fine-tunes that checkpoint with **PT2E QAT** and
   lowers it to an **int8 XNNPACK `.pte`** for on-device CPU/ARM inference.
4. **Export** turns a checkpoint into an ONNX or ExecuTorch artifact that accepts a raw
   grayscale image of any size and outputs character probabilities.

**Inference.**
Input: an image `(B, C, H, W)` with `C = 1` (grayscale) or `C = 3`
(RGB), any size, any range. The model resizes and reduces colour to one channel internally.
Output: probabilities `(B, num_classes)` aligned with `labels.txt`.


## Apps that use this model

| name | android | iOS | Linux | MacOS | Windows | Web |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DaKanji | ✅ | ✅ | ✅ | ✅ | ✅ |   |

## Usage in your software

I put lots of effort and time into developing this model and hope that it can be used in many apps.
If you decide to use this machine learning model please give me credit like:
`Character recognition powered by machine learning from CaptainDario (DaAppLab)`
It would also be nice if you open an issue and tell me that you are using this model.
Than I would add your software to the [apps section](#apps-which-use-this-model)

## Credits

Training data kindly provided by the
[ETL Character Database](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database) and
[the KanjiVG dataset](https://kanjivg.tagaini.net/). Architecture based on
[EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946).

## License

[MIT](LICENSE) © CaptainDario
