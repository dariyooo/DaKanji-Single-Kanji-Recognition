"""Training notebook (Marimo): interactive experimentation.

Loads a base config from ``configs/`` and lets you tweak the key knobs with widgets.
The effective config (base + overrides) is logged to MLflow, so interactive runs are
still tracked. For systematic searches use ``scripts/architecture_search.py`` instead.

    uv run marimo edit notebooks/train.py
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # DaKanji: Single Character CNN · Training

        Pick a base config, tweak the knobs, press **Train**. Preprocessing
        (resize + normalize) is inside the model; the effective config is logged to MLflow.
        """
    )
    return


@app.cell
def _():
    import copy

    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    import torch

    from char_recognition.config.loader import list_configs, load_config, resolve_device
    from char_recognition.engine.runner import train_from_config
    from char_recognition.models.registry import available_backbones
    from char_recognition.paths import JP_FONT

    return (
        JP_FONT,
        available_backbones,
        copy,
        fm,
        list_configs,
        load_config,
        plt,
        resolve_device,
        torch,
        train_from_config,
    )


@app.cell
def _(list_configs, mo):
    config_files = [str(p) for p in list_configs() if "grid" not in p.name]
    default_config = next((c for c in config_files if "synthetic" in c), config_files[0])
    config_picker = mo.ui.dropdown(config_files, value=default_config, label="Base config")
    config_picker
    return (config_picker,)


@app.cell
def _(config_picker, load_config):
    base_cfg = load_config(config_picker.value)
    return (base_cfg,)


@app.cell
def _(available_backbones, base_cfg, mo):
    size_options = sorted({"48", "64", "96", "128", str(base_cfg.data.image_size[0])}, key=int)
    lr_options = sorted({"0.0005", "0.001", "0.002", "0.003", str(base_cfg.optim.lr)})
    overrides = mo.ui.dictionary(
        {
            "backbone": mo.ui.dropdown(available_backbones(), value=base_cfg.model.name, label="Backbone"),
            "image_size": mo.ui.dropdown(size_options, value=str(base_cfg.data.image_size[0]), label="Input size"),
            "batch_size": mo.ui.slider(8, 512, value=base_cfg.data.batch_size, step=8, label="Batch size"),
            "epochs": mo.ui.slider(1, 100, value=base_cfg.optim.epochs, label="Epochs"),
            "lr": mo.ui.dropdown(lr_options, value=str(base_cfg.optim.lr), label="Learning rate"),
            "mixup": mo.ui.switch(base_cfg.augment.mix_p > 0, label="MixUp / CutMix"),
        }
    )
    overrides
    return (overrides,)


@app.cell
def _(base_cfg, copy, mo, overrides, resolve_device):
    cfg = copy.deepcopy(base_cfg)
    size = int(overrides["image_size"].value)
    cfg.model.name = overrides["backbone"].value
    cfg.data.image_size = (size, size)
    cfg.data.batch_size = overrides["batch_size"].value
    cfg.data.num_workers = 0  # keep the reactive notebook responsive
    cfg.optim.epochs = overrides["epochs"].value
    cfg.optim.lr = float(overrides["lr"].value)
    cfg.augment.mix_p = 0.5 if overrides["mixup"].value else 0.0

    device = resolve_device(cfg.device)
    mo.md(f"**Device:** `{device}`\n\n```\n{cfg}\n```")
    return cfg, device


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="▶ Train")
    train_button
    return (train_button,)


@app.cell
def _(cfg, device, train_button, train_from_config):
    result = train_from_config(cfg, device) if train_button.value else None
    return (result,)


@app.cell
def _(mo, plt, result):
    if result is None:
        curves = mo.md("*Press ▶ Train to start.*")
    else:
        history = result.history
        curve_fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
        epochs_axis = range(1, len(history.train_loss) + 1)
        ax_loss.plot(epochs_axis, history.train_loss, label="train")
        ax_loss.plot(epochs_axis, history.val_loss, label="val")
        ax_loss.set(title="Loss", xlabel="epoch")
        ax_loss.legend()
        ax_acc.plot(epochs_axis, history.train_acc, label="train")
        ax_acc.plot(epochs_axis, history.val_acc, label="val")
        ax_acc.set(title="Accuracy", xlabel="epoch")
        ax_acc.legend()
        curve_fig.tight_layout()
        curves = mo.as_html(curve_fig)
    curves
    return


@app.cell
def _(mo):
    mo.md("## Predictions: 10×10 grid")
    return


@app.cell
def _(JP_FONT, fm, mo, plt, result, torch):
    if result is None:
        grid = mo.md("*Train first to see predictions.*")
    else:
        jp_font = fm.FontProperties(fname=str(JP_FONT)) if JP_FONT.exists() else None

        collected_images, collected_targets = [], []
        for batch_images, batch_targets in result.val_loader:
            collected_images.append(batch_images)
            collected_targets.append(batch_targets)
            if sum(t.shape[0] for t in collected_images) >= 100:
                break
        images = torch.cat(collected_images)[:100]
        targets = torch.cat(collected_targets)[:100]

        device_ = next(result.model.parameters()).device
        result.model.eval()
        with torch.no_grad():
            probs = torch.softmax(result.model(images.to(device_)), dim=1)
        confidence, predicted = probs.max(dim=1)

        grid_fig, grid_axes = plt.subplots(10, 10, figsize=(14, 14))
        for idx, ax in enumerate(grid_axes.flat):
            ax.imshow(images[idx, 0].cpu().numpy(), cmap="gray")
            ax.axis("off")
            is_correct = predicted[idx].item() == targets[idx].item()
            ax.set_title(
                f"{result.labels[predicted[idx]]} {confidence[idx]:.2f}",
                fontproperties=jp_font,
                color="green" if is_correct else "red",
                fontsize=8,
            )
        grid_fig.tight_layout()
        grid = mo.as_html(grid_fig)
    grid
    return


if __name__ == "__main__":
    app.run()
