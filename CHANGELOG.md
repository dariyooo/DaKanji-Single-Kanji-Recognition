# DaKanjiRecognizer - Machine Learning : changelog

## v2.0

Rewrite: TensorFlow/Keras + Jupyter -> PyTorch + Marimo, managed by uv (Ruff, mypy).

- Ported to PyTorch with resize/normalize **inside the model** (export needs no in-app
  preprocessing); custom AMP train loop, manual checkpointing, MLflow logging.
- Plug-and-play model registry (EfficientNet-Lite B0 port + timm reference, MobileNetV3,
  tiny baseline) with configurable input size.
- Augmentation on `torchvision.transforms.v2` (MixUp, CutMix, shear, sharpness, cutout).
- Generic folder data loader (labels + folder) + synthetic test data; TOML configs
  split by type and composed by run configs.
- Two-stage int8 quantization: train fp32, then a **PT2E QAT** fine-tune (conv+linear)
  lowered to an **XNNPACK ExecuTorch `.pte`** for on-device CPU/ARM. Config-driven
  grid-search sweeps model × input size; ONNX and ExecuTorch export keep dynamic
  shapes (TFLite removed).
- Interactive Marimo training notebook + headless CLI; end-to-end tests.
- Purged the data-generation notebooks and TensorFlow-era files.

-------------------------------------------------------------------------

## v1.2

new Features:

- recognize:

  - all 漢字 from 漢字検定
  - ひらがな (also historical ones: ゑ, etc.)
  - カタカナ (also historical ones: ヱ, etc.)

Changes:

- handle class imbalance better
- moved fonts into root directory

-------------------------------------------------------------------------

## v 1.1

changes:

- moved from jupyter notebook to jupyter lab
- multi processing for loading the data
- data generator for feeding batches to the CNN
  - multi processed
  - image augmentation

-------------------------------------------------------------------------

## v 1.0

features:

- recognize ~3000 kanji characters
  