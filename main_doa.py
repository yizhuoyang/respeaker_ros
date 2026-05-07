import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from dataloader.ssl_dataset import SingleStepDataset
from network.audionet.ssl_net import SSLNet, SSLNet_depth, SSLNet_DOA, SSLNet_depth_DOA,SSLNet_depth_DOA_res,SSLNet_DOA_res
from utlis.loss import _neg_loss
from model_training.train_doa import train_one_epoch, validate 


TRAIN_META = "/home/Disk/sound-space/ssl_data_semantic/train"
VAL_META   = "/home/Disk/sound-space/ssl_data_semantic/val"

BATCH_SIZE   = 256
NUM_EPOCHS   = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS  = 4

LOG_DIR  = "runs/ssl_doa_r_audio_depth_ipd_tune"
SAVE_DIR = "/home/Disk/yyz/sound-spaces/weights/ssl_doa_r_audio_depth_ipd_tune"
eval_epoch  = 10
use_compress = False
mode = 'doa_distance'
audio_feat = 'ipd'
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========================
    # 1. Dataset / DataLoader
    # ========================
    train_dataset = SingleStepDataset(
        root_dir=TRAIN_META,
        use_compress=use_compress,
        mode=mode,
        audio_feat = audio_feat
        
        
    )
    val_dataset = SingleStepDataset(
        root_dir=VAL_META,
        use_compress=use_compress,
        mode=mode,
        audio_feat = audio_feat

    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ========================
    # 2. Model
    # ========================
    # model = SSLNet_DOA(use_compress=use_compress)
    model = SSLNet_depth_DOA(use_compress=use_compress)
    # CKPT_PATH = "/media/kemove/data/sound-spaces/data/models/savi_final/best_val.pth"
    CKPT_PATH = "/home/Disk/yyz/sound-spaces/weights/ssl_doa_r_audio_depth_ipd/last_model.pth"
    # if CKPT_PATH is not None and os.path.exists(CKPT_PATH):
    #     ckpt = torch.load(CKPT_PATH, map_location="cpu")
    #     if "audiogoal_predictor" in ckpt:
    #         model.load_state_dict(ckpt["audiogoal_predictor"], strict=False)
    #         print(f"[INFO] loaded ckpt: {CKPT_PATH}")
    #     else:
    #         model.load_state_dict(ckpt, strict=False)
    #         print(f"[INFO] loaded ckpt (raw state_dict): {CKPT_PATH}")
    # else:
    #     print("[WARN] ckpt not loaded (path is None or not exists).")
    if CKPT_PATH is None:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt)
    model = model.to(device)

    # for p in model.spec_encoder.parameters():
    #     p.requires_grad = False

    #     model.spec_encoder.eval()  

    # optimizer = torch.optim.Adam(
    #     list(model.doa_head.parameters()) + list(model.distance_head.parameters()),
    #     lr=LR,
    #     weight_decay=WEIGHT_DECAY,
    # )

    # ========================
    # 3. Optimizer + Scheduler + TB writer
    # ========================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS, 
        eta_min=1e-6,      
    )

    criterion = nn.MSELoss()
    
    writer = SummaryWriter(LOG_DIR)

    # ========================
    # 4. Training loop
    # ========================
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.6e}")
        writer.add_scalar("lr", current_lr, epoch)

        train_loss, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
        )

        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            writer=writer,
            max_images=3,
        )

        print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")


        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> Saved best model to {ckpt_path}, val_loss={best_val_loss:.4f}")

        last_ckpt = os.path.join(SAVE_DIR, "last_model.pth")
        torch.save(model.state_dict(), last_ckpt)

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
