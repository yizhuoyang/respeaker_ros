import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def soft_cross_entropy(logits, target):
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_prob = F.log_softmax(logits, dim=1)
    return -(target * log_prob).sum(dim=1).mean()


def masked_soft_cross_entropy(logits, target, mask):
    if mask is None:
        return soft_cross_entropy(logits, target)
    mask = mask.bool()
    if not torch.any(mask):
        return logits.sum() * 0.0
    return soft_cross_entropy(logits[mask], target[mask])


def combined_loss(
    logits_doa,
    logits_dist,
    target_doa,
    target_dist,
    distance_weight=0.5,
    class_logits=None,
    class_target=None,
    class_weight=None,
    classification_weight=0.0,
    has_doa=None,
    classification_only=False,
):
    loss_doa = masked_soft_cross_entropy(logits_doa, target_doa, has_doa)
    loss_dist = masked_soft_cross_entropy(logits_dist, target_dist, has_doa)
    if class_logits is None or class_target is None or classification_weight <= 0:
        loss_cls = logits_doa.sum() * 0.0
    else:
        loss_cls = F.cross_entropy(class_logits, class_target.long(), weight=class_weight)
    if classification_only:
        total = loss_cls
        return total, loss_doa, loss_dist, loss_cls
    total = loss_doa + distance_weight * loss_dist + classification_weight * loss_cls
    return total, loss_doa, loss_dist, loss_cls


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    writer=None,
    global_step=0,
    criterion=None,
    distance_weight=0,
    classification_weight=0.0,
    class_weights=None,
    classification_only=False,
    gate_doa_by_pred_class=False,
    signal_class_id=2,
    freeze_classifier_eval=False,
):
    model.train()
    if freeze_classifier_eval:
        set_classifier_gate_modules_eval(model)
    running_loss = 0.0
    running_doa = 0.0
    running_dist = 0.0
    running_cls = 0.0
    running_cls_correct = 0
    running_cls_total = 0
    running_gate_active = 0
    running_gate_total = 0
    n_samples = 0
    if class_weights is not None:
        class_weights = class_weights.to(device)

    for batch in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
        depth = batch["depth"].to(device, non_blocking=True)
        spectrogram = batch["spectrogram"].to(device, non_blocking=True)
        target_doa = batch["doa_map"].to(device, non_blocking=True)
        target_dist = batch["distant_map"].to(device, non_blocking=True)
        class_target = batch.get("class_label")
        has_doa = batch.get("has_doa")
        if class_target is not None:
            class_target = class_target.to(device, non_blocking=True)
        if has_doa is not None:
            has_doa = has_doa.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(spectrogram, depth)
        logits_doa, logits_dist = outputs[:2]
        class_logits = outputs[2] if len(outputs) > 2 else None
        loss_mask = has_doa
        if gate_doa_by_pred_class and class_logits is not None:
            pred_signal = class_logits.detach().argmax(dim=1) == int(signal_class_id)
            loss_mask = pred_signal if loss_mask is None else (loss_mask.bool() & pred_signal)
        if loss_mask is not None:
            running_gate_active += int(loss_mask.bool().sum().item())
            running_gate_total += int(loss_mask.numel())
        loss, loss_doa, loss_dist, loss_cls = combined_loss(
            logits_doa,
            logits_dist,
            target_doa,
            target_dist,
            distance_weight=distance_weight,
            class_logits=class_logits,
            class_target=class_target,
            class_weight=class_weights,
            classification_weight=classification_weight,
            has_doa=loss_mask,
            classification_only=classification_only,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        batch_size = spectrogram.size(0)
        running_loss += loss.item() * batch_size
        running_doa += loss_doa.item() * batch_size
        running_dist += loss_dist.item() * batch_size
        running_cls += loss_cls.item() * batch_size
        if class_logits is not None and class_target is not None:
            running_cls_correct += (class_logits.argmax(dim=1) == class_target).sum().item()
            running_cls_total += batch_size
        n_samples += batch_size

        if writer is not None:
            writer.add_scalar("Loss/train_step_total", loss.item(), global_step)
            writer.add_scalar("Loss/train_step_doa", loss_doa.item(), global_step)
            writer.add_scalar("Loss/train_step_distance", loss_dist.item(), global_step)
            writer.add_scalar("Loss/train_step_class", loss_cls.item(), global_step)
        global_step += 1

    metrics = {
        "loss": running_loss / max(n_samples, 1),
        "loss_doa": running_doa / max(n_samples, 1),
        "loss_distance": running_dist / max(n_samples, 1),
        "loss_class": running_cls / max(n_samples, 1),
        "class_acc": running_cls_correct / max(running_cls_total, 1),
        "gate_active_ratio": running_gate_active / max(running_gate_total, 1),
    }
    if writer is not None:
        writer.add_scalar("Loss/train_epoch_total", metrics["loss"], epoch)
        writer.add_scalar("Loss/train_epoch_doa", metrics["loss_doa"], epoch)
        writer.add_scalar("Loss/train_epoch_distance", metrics["loss_distance"], epoch)
        writer.add_scalar("Loss/train_epoch_class", metrics["loss_class"], epoch)
        writer.add_scalar("Metric/train_class_acc", metrics["class_acc"], epoch)
        writer.add_scalar("Metric/train_gate_active_ratio", metrics["gate_active_ratio"], epoch)
    print(f"Train gate active ratio: {metrics['gate_active_ratio']:.4f}")
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
    distance_weight=0,
    classification_weight=0.0,
    class_weights=None,
    classification_only=False,
    gate_doa_by_pred_class=False,
    signal_class_id=2,
):
    model.eval()
    running_loss = 0.0
    running_doa = 0.0
    running_dist = 0.0
    running_cls = 0.0
    running_cls_correct = 0
    running_cls_total = 0
    running_gate_active = 0
    running_gate_total = 0
    n_samples = 0
    first_batch_for_plot = None
    if class_weights is not None:
        class_weights = class_weights.to(device)

    for batch in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
        depth = batch["depth"].to(device, non_blocking=True)
        spectrogram = batch["spectrogram"].to(device, non_blocking=True)
        target_doa = batch["doa_map"].to(device, non_blocking=True)
        target_dist = batch["distant_map"].to(device, non_blocking=True)
        class_target = batch.get("class_label")
        has_doa = batch.get("has_doa")
        if class_target is not None:
            class_target = class_target.to(device, non_blocking=True)
        if has_doa is not None:
            has_doa = has_doa.to(device, non_blocking=True)

        outputs = model(spectrogram, depth)
        logits_doa, logits_dist = outputs[:2]
        class_logits = outputs[2] if len(outputs) > 2 else None
        loss_mask = has_doa
        if gate_doa_by_pred_class and class_logits is not None:
            pred_signal = class_logits.detach().argmax(dim=1) == int(signal_class_id)
            loss_mask = pred_signal if loss_mask is None else (loss_mask.bool() & pred_signal)
        if loss_mask is not None:
            running_gate_active += int(loss_mask.bool().sum().item())
            running_gate_total += int(loss_mask.numel())
        loss, loss_doa, loss_dist, loss_cls = combined_loss(
            logits_doa,
            logits_dist,
            target_doa,
            target_dist,
            distance_weight=distance_weight,
            class_logits=class_logits,
            class_target=class_target,
            class_weight=class_weights,
            classification_weight=classification_weight,
            has_doa=loss_mask,
            classification_only=classification_only,
        )

        batch_size = spectrogram.size(0)
        running_loss += loss.item() * batch_size
        running_doa += loss_doa.item() * batch_size
        running_dist += loss_dist.item() * batch_size
        running_cls += loss_cls.item() * batch_size
        if class_logits is not None and class_target is not None:
            running_cls_correct += (class_logits.argmax(dim=1) == class_target).sum().item()
            running_cls_total += batch_size
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
        "loss_class": running_cls / max(n_samples, 1),
        "class_acc": running_cls_correct / max(running_cls_total, 1),
        "gate_active_ratio": running_gate_active / max(running_gate_total, 1),
    }
    if writer is not None:
        writer.add_scalar("Loss/val_epoch_total", metrics["loss"], epoch)
        writer.add_scalar("Loss/val_epoch_doa", metrics["loss_doa"], epoch)
        writer.add_scalar("Loss/val_epoch_distance", metrics["loss_distance"], epoch)
        writer.add_scalar("Loss/val_epoch_class", metrics["loss_class"], epoch)
        writer.add_scalar("Metric/val_class_acc", metrics["class_acc"], epoch)
        writer.add_scalar("Metric/val_gate_active_ratio", metrics["gate_active_ratio"], epoch)
        if first_batch_for_plot is not None and (epoch == 1 or epoch % 5 == 0):
            add_distribution_figures(writer, first_batch_for_plot, epoch, max_images=max_images)
    print(f"Val gate active ratio: {metrics['gate_active_ratio']:.4f}")
    return metrics["loss"]


def set_classifier_gate_modules_eval(model):
    for name in [
        "spec_encoder",
        "depth_encoder",
        "film_gamma",
        "film_beta",
        "fusion_fc",
        "class_head",
    ]:
        module = getattr(model, name, None)
        if module is not None:
            module.eval()


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
