import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def soft_cross_entropy(logits, target):
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_prob = F.log_softmax(logits, dim=1)
    return -(target * log_prob).sum(dim=1).mean()


def combined_loss(logits_doa, logits_dist, target_doa, target_dist, distance_weight=0.5):
    loss_doa = soft_cross_entropy(logits_doa, target_doa)
    loss_dist = soft_cross_entropy(logits_dist, target_dist)
    return loss_doa + distance_weight * loss_dist, loss_doa, loss_dist


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    writer=None,
    global_step=0,
    criterion=None,
    distance_weight=0.5,
):
    model.train()
    running_loss = 0.0
    running_doa = 0.0
    running_dist = 0.0
    n_samples = 0

    for batch in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
        depth = batch["depth"].to(device, non_blocking=True)
        spectrogram = batch["spectrogram"].to(device, non_blocking=True)
        target_doa = batch["doa_map"].to(device, non_blocking=True)
        target_dist = batch["distant_map"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits_doa, logits_dist = model(spectrogram, depth)
        loss, loss_doa, loss_dist = combined_loss(
            logits_doa,
            logits_dist,
            target_doa,
            target_dist,
            distance_weight=distance_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        batch_size = spectrogram.size(0)
        running_loss += loss.item() * batch_size
        running_doa += loss_doa.item() * batch_size
        running_dist += loss_dist.item() * batch_size
        n_samples += batch_size

        if writer is not None:
            writer.add_scalar("Loss/train_step_total", loss.item(), global_step)
            writer.add_scalar("Loss/train_step_doa", loss_doa.item(), global_step)
            writer.add_scalar("Loss/train_step_distance", loss_dist.item(), global_step)
        global_step += 1

    metrics = {
        "loss": running_loss / max(n_samples, 1),
        "loss_doa": running_doa / max(n_samples, 1),
        "loss_distance": running_dist / max(n_samples, 1),
    }
    if writer is not None:
        writer.add_scalar("Loss/train_epoch_total", metrics["loss"], epoch)
        writer.add_scalar("Loss/train_epoch_doa", metrics["loss_doa"], epoch)
        writer.add_scalar("Loss/train_epoch_distance", metrics["loss_distance"], epoch)
    return metrics["loss"], global_step


@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    epoch,
    writer=None,
    max_images=3,
    criterion=None,
    distance_weight=0.5,
):
    model.eval()
    running_loss = 0.0
    running_doa = 0.0
    running_dist = 0.0
    n_samples = 0
    first_batch_for_plot = None

    for batch in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
        depth = batch["depth"].to(device, non_blocking=True)
        spectrogram = batch["spectrogram"].to(device, non_blocking=True)
        target_doa = batch["doa_map"].to(device, non_blocking=True)
        target_dist = batch["distant_map"].to(device, non_blocking=True)

        logits_doa, logits_dist = model(spectrogram, depth)
        loss, loss_doa, loss_dist = combined_loss(
            logits_doa,
            logits_dist,
            target_doa,
            target_dist,
            distance_weight=distance_weight,
        )

        batch_size = spectrogram.size(0)
        running_loss += loss.item() * batch_size
        running_doa += loss_doa.item() * batch_size
        running_dist += loss_dist.item() * batch_size
        n_samples += batch_size

        if first_batch_for_plot is None:
            first_batch_for_plot = (
                torch.softmax(logits_doa, dim=1).detach().cpu(),
                target_doa.detach().cpu(),
                torch.softmax(logits_dist, dim=1).detach().cpu(),
                target_dist.detach().cpu(),
            )

    metrics = {
        "loss": running_loss / max(n_samples, 1),
        "loss_doa": running_doa / max(n_samples, 1),
        "loss_distance": running_dist / max(n_samples, 1),
    }
    if writer is not None:
        writer.add_scalar("Loss/val_epoch_total", metrics["loss"], epoch)
        writer.add_scalar("Loss/val_epoch_doa", metrics["loss_doa"], epoch)
        writer.add_scalar("Loss/val_epoch_distance", metrics["loss_distance"], epoch)
        if first_batch_for_plot is not None and (epoch == 1 or epoch % 5 == 0):
            add_distribution_figures(writer, first_batch_for_plot, epoch, max_images=max_images)
    return metrics["loss"]


def add_distribution_figures(writer, batch_tensors, epoch, max_images=3):
    pred_doa, target_doa, pred_dist, target_dist = batch_tensors
    n_show = min(max_images, pred_doa.size(0))

    fig, axes = plt.subplots(n_show, 1, figsize=(7, 2.2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    x = np.arange(pred_doa.size(1))
    for i in range(n_show):
        axes[i].plot(x, target_doa[i].numpy(), label="GT")
        axes[i].plot(x, pred_doa[i].numpy(), label="Pred", linestyle="--")
        axes[i].grid(True)
    axes[0].set_title("DOA distribution")
    axes[-1].set_xlabel("angle bin")
    axes[0].legend()
    writer.add_figure("Val/DOA_GT_vs_Pred", fig, global_step=epoch)
    plt.close(fig)

    fig, axes = plt.subplots(n_show, 1, figsize=(7, 2.2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    x = np.arange(pred_dist.size(1))
    for i in range(n_show):
        axes[i].plot(x, target_dist[i].numpy(), label="GT")
        axes[i].plot(x, pred_dist[i].numpy(), label="Pred", linestyle="--")
        axes[i].grid(True)
    axes[0].set_title("Distance distribution")
    axes[-1].set_xlabel("distance bin")
    axes[0].legend()
    writer.add_figure("Val/Distance_GT_vs_Pred", fig, global_step=epoch)
    plt.close(fig)
