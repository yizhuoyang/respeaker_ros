import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils  
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    epoch, writer=None, global_step=0):
    
    model.train()
    # model.spec_encoder.eval()  
    running_loss = 0.0
    n_samples = 0

    for batch in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
        depth       = batch["depth"].to(device)
        spectrogram = batch["spectrogram"].to(device)
        target_hm   = batch["doa_map"].to(device)
        target_dis = batch["distant_map"].to(device)

        optimizer.zero_grad()

        pred_hm,pred_dis = model(spectrogram, depth)  

        # pred_hm_log  = F.log_softmax(pred_hm,  dim=1)  # [B, doa_bins]
        # pred_dis_log = F.log_softmax(pred_dis, dim=1)  # [B, r_bins]
        # loss_doa = F.kl_div(pred_hm_log, target_hm, reduction='batchmean')
        # loss_r   = F.kl_div(pred_dis_log, target_dis,   reduction='batchmean')
        # loss = loss_doa
        loss = criterion(pred_hm, target_hm)+criterion(pred_dis, target_dis)*0.2

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
             epoch, writer=None, max_images=3):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Val {epoch}", leave=False)):
            depth       = batch["depth"].to(device)
            spectrogram = batch["spectrogram"].to(device)
            target_hm   = batch["doa_map"].to(device)
            target_dis = batch["distant_map"].to(device)

            pred_hm,pred_dis = model(spectrogram, depth)  

            # pred_hm_log  = F.log_softmax(pred_hm,  dim=1)  # [B, doa_bins]
            # pred_dis_log = F.log_softmax(pred_dis, dim=1)  # [B, r_bins]
            # loss_doa = F.kl_div(pred_hm_log, target_hm, reduction='batchmean')
            # loss_r   = F.kl_div(pred_dis_log, target_dis,   reduction='batchmean')
            # loss = loss_doa
            loss = criterion(pred_hm, target_hm)+criterion(pred_dis, target_dis)*0.2

            batch_size = depth.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            # if writer is not None and batch_idx == 0 and epoch%5==0:
            #     prob_pred = pred_hm
            #     pred_dis  = pred_dis
            #     B = pred_dis.size(0)
            #     n_show = min(B, max_images)

            #     pred_d_vis = pred_dis[:n_show].detach().cpu().numpy()    # [B,R]
            #     gt_d_vis   = target_dis[:n_show].detach().cpu().numpy()  # [B,R]

            #     pred_doa_vis = prob_pred[:n_show].detach().cpu().numpy()    # [B,R]
            #     gt_doa_vis   = target_hm[:n_show].detach().cpu().numpy()  # [B,R]


            #     num_bins = pred_d_vis.shape[1]
            #     x = np.arange(num_bins)

            #     fig, axes = plt.subplots(n_show, 1, figsize=(6, 2*n_show), sharex=True)
            #     if n_show == 1:
            #         axes = [axes]

            #     for i in range(n_show):
            #         ax = axes[i]
            #         ax.plot(x, gt_d_vis[i], label="GT")
            #         ax.plot(x, pred_d_vis[i], label="Pred", linestyle="--")
            #         ax.set_ylabel(f"sample {i}")
            #         ax.grid(True)
            #     axes[0].set_title("Distance distribution")
            #     axes[-1].set_xlabel("bin index")
            #     axes[0].legend()

            #     writer.add_figure("R_curve/GT_vs_Pred", fig, global_step=epoch)
            #     plt.close(fig)

            #     num_bins = pred_doa_vis.shape[1]
            #     x = np.arange(num_bins)
            #     fig, axes = plt.subplots(n_show, 1, figsize=(6, 2*n_show), sharex=True)
            #     if n_show == 1:
            #         axes = [axes]

            #     for i in range(n_show):
            #         ax = axes[i]
            #         ax.plot(x, gt_doa_vis[i], label="GT")
            #         ax.plot(x, pred_doa_vis[i], label="Pred", linestyle="--")
            #         ax.set_ylabel(f"sample {i}")
            #         ax.grid(True)
            #     axes[0].set_title("DOA distribution")
            #     axes[-1].set_xlabel("bin index")
            #     axes[0].legend()

            #     writer.add_figure("DOA_curve/GT_vs_Pred", fig, global_step=epoch)
            #     plt.close(fig)


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

