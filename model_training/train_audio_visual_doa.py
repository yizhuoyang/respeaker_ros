import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def distribution_mse_loss(logits, target):
    pred = torch.sigmoid(logits)
    return F.mse_loss(pred, target)


def combined_loss(doa_logits, distance_logits, doa_target, distance_target, distance_weight=0.5):
    loss_doa = distribution_mse_loss(doa_logits, doa_target)
    loss_distance = distribution_mse_loss(distance_logits, distance_target)
    loss = loss_doa + distance_weight * loss_distance
    return loss, loss_doa, loss_distance


def distribution_argmax_error(pred_probs, target_probs, circular=False):
    pred_idx = pred_probs.argmax(dim=1)
    target_idx = target_probs.argmax(dim=1)
    error = (pred_idx - target_idx).abs().float()
    if circular:
        bins = pred_probs.size(1)
        error = torch.minimum(error, bins - error)
    return error.mean().item()


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    writer=None,
    global_step=0,
    distance_weight=0.5,
):
    model.train()
    totals = {"loss": 0.0, "loss_doa": 0.0, "loss_distance": 0.0}
    n_samples = 0
    doa_bin_errors = []
    distance_bin_errors = []

    for batch in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
        spectrogram = batch["spectrogram"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        doa_target = batch["doa_map"].to(device, non_blocking=True)
        distance_target = batch["distance_map"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        doa_logits, distance_logits = model(spectrogram, image=image, depth=depth)
        loss, loss_doa, loss_distance = combined_loss(
            doa_logits,
            distance_logits,
            doa_target,
            distance_target,
            distance_weight=distance_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        with torch.no_grad():
            doa_prob = torch.sigmoid(doa_logits)
            distance_prob = torch.sigmoid(distance_logits)
            doa_bin_errors.append(distribution_argmax_error(doa_prob, doa_target, circular=True))
            distance_bin_errors.append(distribution_argmax_error(distance_prob, distance_target, circular=False))

        batch_size = spectrogram.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["loss_doa"] += loss_doa.item() * batch_size
        totals["loss_distance"] += loss_distance.item() * batch_size
        n_samples += batch_size

        if writer is not None:
            writer.add_scalar("Loss/train_step_total", loss.item(), global_step)
            writer.add_scalar("Loss/train_step_doa", loss_doa.item(), global_step)
            writer.add_scalar("Loss/train_step_distance", loss_distance.item(), global_step)
        global_step += 1

    metrics = {key: value / max(n_samples, 1) for key, value in totals.items()}
    metrics["doa_bin_error"] = float(np.mean(doa_bin_errors)) if doa_bin_errors else 0.0
    metrics["distance_bin_error"] = float(np.mean(distance_bin_errors)) if distance_bin_errors else 0.0
    if writer is not None:
        writer.add_scalar("Loss/train_epoch_total", metrics["loss"], epoch)
        writer.add_scalar("Loss/train_epoch_doa", metrics["loss_doa"], epoch)
        writer.add_scalar("Loss/train_epoch_distance", metrics["loss_distance"], epoch)
        writer.add_scalar("Metric/train_doa_bin_error", metrics["doa_bin_error"], epoch)
        writer.add_scalar("Metric/train_distance_bin_error", metrics["distance_bin_error"], epoch)
    return metrics, global_step


@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    epoch,
    writer=None,
    distance_weight=0.5,
    max_plots=3,
):
    model.eval()
    totals = {"loss": 0.0, "loss_doa": 0.0, "loss_distance": 0.0}
    n_samples = 0
    first_batch_for_plot = None
    doa_bin_errors = []
    distance_bin_errors = []

    for batch in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
        spectrogram = batch["spectrogram"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        doa_target = batch["doa_map"].to(device, non_blocking=True)
        distance_target = batch["distance_map"].to(device, non_blocking=True)

        doa_logits, distance_logits = model(spectrogram, image=image, depth=depth)
        loss, loss_doa, loss_distance = combined_loss(
            doa_logits,
            distance_logits,
            doa_target,
            distance_target,
            distance_weight=distance_weight,
        )

        doa_prob = torch.sigmoid(doa_logits)
        distance_prob = torch.sigmoid(distance_logits)
        doa_bin_errors.append(distribution_argmax_error(doa_prob, doa_target, circular=True))
        distance_bin_errors.append(distribution_argmax_error(distance_prob, distance_target, circular=False))

        batch_size = spectrogram.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["loss_doa"] += loss_doa.item() * batch_size
        totals["loss_distance"] += loss_distance.item() * batch_size
        n_samples += batch_size

        if first_batch_for_plot is None:
            first_batch_for_plot = (
                doa_prob.detach().cpu(),
                doa_target.detach().cpu(),
                distance_prob.detach().cpu(),
                distance_target.detach().cpu(),
            )

    metrics = {key: value / max(n_samples, 1) for key, value in totals.items()}
    metrics["doa_bin_error"] = float(np.mean(doa_bin_errors)) if doa_bin_errors else 0.0
    metrics["distance_bin_error"] = float(np.mean(distance_bin_errors)) if distance_bin_errors else 0.0

    if writer is not None:
        writer.add_scalar("Loss/val_epoch_total", metrics["loss"], epoch)
        writer.add_scalar("Loss/val_epoch_doa", metrics["loss_doa"], epoch)
        writer.add_scalar("Loss/val_epoch_distance", metrics["loss_distance"], epoch)
        writer.add_scalar("Metric/val_doa_bin_error", metrics["doa_bin_error"], epoch)
        writer.add_scalar("Metric/val_distance_bin_error", metrics["distance_bin_error"], epoch)
        if first_batch_for_plot is not None and (epoch == 1 or epoch % 5 == 0):
            add_distribution_figures(writer, first_batch_for_plot, epoch, max_plots=max_plots)

    return metrics


def add_distribution_figures(writer, batch_tensors, epoch, max_plots=3):
    pred_doa, target_doa, pred_distance, target_distance = batch_tensors
    n_show = min(max_plots, pred_doa.size(0))

    fig, axes = plt.subplots(n_show, 1, figsize=(7, 2.2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    x = np.arange(pred_doa.size(1))
    for index in range(n_show):
        axes[index].plot(x, target_doa[index].numpy(), label="GT")
        axes[index].plot(x, pred_doa[index].numpy(), label="Pred", linestyle="--")
        axes[index].grid(True)
    axes[0].set_title("DOA distribution")
    axes[-1].set_xlabel("angle bin")
    axes[0].legend()
    writer.add_figure("Val/DOA_GT_vs_Pred", fig, global_step=epoch)
    plt.close(fig)

    fig, axes = plt.subplots(n_show, 1, figsize=(7, 2.2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    x = np.arange(pred_distance.size(1))
    for index in range(n_show):
        axes[index].plot(x, target_distance[index].numpy(), label="GT")
        axes[index].plot(x, pred_distance[index].numpy(), label="Pred", linestyle="--")
        axes[index].grid(True)
    axes[0].set_title("Distance distribution")
    axes[-1].set_xlabel("distance bin")
    axes[0].legend()
    writer.add_figure("Val/Distance_GT_vs_Pred", fig, global_step=epoch)
    plt.close(fig)
