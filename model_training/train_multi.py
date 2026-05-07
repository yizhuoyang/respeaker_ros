import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils  

def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    epoch, writer=None, global_step=0,extra_modal='depth'):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
        spectrogram = batch["spectrogram"].to(device)
        target_hm   = batch["heatmap"].to(device)
        rel_pose    = batch["rel_pose"].to(device)
        if extra_modal=='depth':
            depth       = batch["depth"].to(device)
        elif extra_modal=='rgb':
            depth       = batch["rgb"].to(device)
        elif extra_modal=='ego':   
            depth       = batch["ego_map"].to(device)
        else:
            depth  = batch["ego_map"].to(device)
        optimizer.zero_grad()

        pred_hm = model(spectrogram, rel_pose)   # (B, 1, 64, 64), logits
        loss = criterion(pred_hm, target_hm)

        loss.backward()
        optimizer.step()

        batch_size = depth.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        # per-step scalar logging
        if writer is not None:
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
        global_step += 1

    epoch_loss = running_loss / max(n_samples, 1)

    # per-epoch scalar logging
    if writer is not None:
        writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)

    return epoch_loss, global_step


def validate(model, val_loader, criterion, device,
             epoch, writer=None, max_images=3,extra_modal='depth'):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Val {epoch}", leave=False)):
            spectrogram = batch["spectrogram"].to(device)
            target_hm   = batch["heatmap"].to(device)
            rel_pose    = batch["rel_pose"].to(device)
            if extra_modal=='depth':
                depth       = batch["depth"].to(device)
            elif extra_modal=='rgb':
                depth       = batch["rgb"].to(device)
            elif extra_modal=='ego':   
                depth       = batch["ego_map"].to(device)
            else:
                depth  = batch["ego_map"].to(device)

            pred_hm = model(spectrogram, rel_pose)          # (B, 1, 64, 64), logits
            loss = criterion(pred_hm, target_hm)

            batch_size = depth.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            if writer is not None and batch_idx == 0 and epoch%5==0:
                prob_pred = torch.sigmoid(pred_hm)   # (B, 1, 64, 64)
                B = prob_pred.size(0)
                n_show = min(B, max_images)
                pred_vis = prob_pred[:n_show].detach().cpu()
                gt_vis   = target_hm[:n_show].detach().cpu()
                # for i in range(n_show):
                #     writer.add_image(f"Pred/heatmap_{i}", pred_vis[i], epoch)
                #     writer.add_image(f"GT/heatmap_{i}",   gt_vis[i],   epoch)
                grid = vutils.make_grid(
                    torch.cat([gt_vis, pred_vis], dim=0), 
                    nrow=n_show 
                )
                writer.add_image("Compare/GT_top_Pred_bottom", grid, epoch)

    epoch_loss = running_loss / max(n_samples, 1)

    if writer is not None:
        writer.add_scalar("Loss/val_epoch", epoch_loss, epoch)

    return epoch_loss


def train_model(model, train_loader, val_loader,
                num_epochs=50,
                lr=1e-4,
                log_dir="runs/av_heatmap"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, writer, global_step
        )

        val_loss = validate(
            model, val_loader, criterion, device,
            epoch, writer, max_images=3
        )

        print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  >> Saved best model with val loss {:.4f}".format(best_val_loss))

    writer.close()

