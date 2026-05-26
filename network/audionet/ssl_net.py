import torch
import torch.nn as nn
from network.audionet.faeture_extraction import SpecEncoderGlobal, DepthResNet18Encoder

class SSLNet_DOA(nn.Module):
    def __init__(
        self,
        spec_out_dim=256,
        depth_out_dim=64,
        fusion_out_dim=256,
        pretrained_depth_encoder=True,
        use_compress=True,        
        num_doa_bins=360,
        num_distance_bins=120,
        audio_in_channels=2,
        num_classes=0,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.spec_encoder = SpecEncoderGlobal(
            in_channels=audio_in_channels,
            channels=(16, 32, 64),
            dropout=0.3,
            out_dim=spec_out_dim,
            use_compress=use_compress
        )

        self.num_doa_bins = num_doa_bins
        self.doa_head = nn.Sequential(
            nn.Linear(spec_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_doa_bins),
        )
        self.num_distance_bins = num_distance_bins
        self.distance_head = nn.Sequential(
            nn.Linear(spec_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_distance_bins),
        )
        self.class_head = None
        if num_classes and num_classes > 0:
            self.class_head = nn.Sequential(
                nn.Linear(spec_out_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )


    def forward(self, spectrogram, depth=None):
        spec_feat = self.spec_encoder(spectrogram)  
        
        doa_logits = self.doa_head(spec_feat)
        distance_logits = self.distance_head(spec_feat)
        if self.class_head is not None:
            class_logits = self.class_head(spec_feat)
            return doa_logits, distance_logits, class_logits
        return doa_logits, distance_logits

class SSLNet_depth_DOA(nn.Module):
    def __init__(
        self,
        spec_out_dim=64,
        depth_out_dim=64,
        fusion_out_dim=256,
        pretrained_depth_encoder=True,
        use_compress=True,
        drop_depth_prob=0.5,  
        num_doa_bins=360,   # 新增：DOA 高斯的维度
        num_distance_bins=120,
        audio_in_channels=2,
        freeze_depth_encoder=False,
        num_classes=0,
    ):
        super().__init__()

        self.drop_depth_prob = drop_depth_prob
        self.num_classes = num_classes

        # ===== Audio encoder =====
        self.spec_encoder = SpecEncoderGlobal(
            in_channels=audio_in_channels,
            channels=(16, 32),
            dropout=0.1,
            out_dim=spec_out_dim,
            use_compress=use_compress,
        )

        # ===== Depth encoder =====
        self.depth_encoder = DepthResNet18Encoder(
            out_dim=depth_out_dim,
            pretrained=pretrained_depth_encoder,
        )


        if freeze_depth_encoder:
            for p in self.depth_encoder.features.parameters():
                p.requires_grad = False

        self.film_gamma = nn.Linear(depth_out_dim, spec_out_dim)
        self.film_beta  = nn.Linear(depth_out_dim, spec_out_dim)

        self.fusion_fc = nn.Sequential(
            nn.Linear(spec_out_dim, fusion_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.out_dim = fusion_out_dim

        # Heatmap decoder: (B, fusion_out_dim) -> (B, 1, 64, 64)
        self.num_doa_bins = num_doa_bins
        self.doa_head = nn.Sequential(
            nn.Linear(fusion_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_doa_bins),
        )

        self.num_distance_bins = num_distance_bins
        self.distance_head = nn.Sequential(
            nn.Linear(fusion_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_distance_bins),
        )
        self.class_head = None
        if num_classes and num_classes > 0:
            self.class_head = nn.Sequential(
                nn.Linear(fusion_out_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

    def forward(self, spectrogram, depth):
        """
        spectrogram: (B, 2, 65/257, 26/101)
        depth:       (B, 3, H, W)
        return:      heatmap (B, 1, 64, 64)
        """
        B = spectrogram.size(0)
        spec_feat = self.spec_encoder(spectrogram)     # (B, spec_out_dim)

        depth_feat = self.depth_encoder(depth)         # (B, depth_out_dim)

        if self.training and torch.rand(1).item() < self.drop_depth_prob:
            depth_feat = torch.zeros_like(depth_feat)

        gamma = self.film_gamma(depth_feat)            # (B, spec_out_dim)
        beta  = self.film_beta(depth_feat)             # (B, spec_out_dim)

        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta  = 0.1 * torch.tanh(beta)

        spec_film = gamma * spec_feat + beta           # (B, spec_out_dim)

        out_feat = self.fusion_fc(spec_film)           # (B, fusion_out_dim)
        
        # DOA head
        doa_logits = self.doa_head(out_feat)
        distance_logits = self.distance_head(out_feat)
        if self.class_head is not None:
            class_logits = self.class_head(out_feat)
            return doa_logits, distance_logits, class_logits
        return doa_logits, distance_logits
