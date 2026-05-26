import torch
import torch.nn as nn

from network.audionet.faeture_extraction import DepthResNet18Encoder, SpecEncoderGlobal


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
        dropout=0.3,
    ):
        super().__init__()

        self.spec_encoder = SpecEncoderGlobal(
            in_channels=audio_in_channels,
            channels=(16, 32, 64),
            dropout=dropout,
            out_dim=spec_out_dim,
            use_compress=use_compress,
        )

        self.num_doa_bins = num_doa_bins
        self.doa_head = nn.Sequential(
            nn.Linear(spec_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_doa_bins),
        )
        self.num_distance_bins = num_distance_bins
        self.distance_head = nn.Sequential(
            nn.Linear(spec_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_distance_bins),
        )

    def forward(self, spectrogram, depth=None):
        spec_feat = self.spec_encoder(spectrogram)
        doa_logits = self.doa_head(spec_feat)
        distance_logits = self.distance_head(spec_feat)
        return doa_logits, distance_logits


class SSLNet_depth_DOA(nn.Module):
    def __init__(
        self,
        spec_out_dim=256,
        depth_out_dim=64,
        fusion_out_dim=256,
        pretrained_depth_encoder=True,
        use_compress=True,
        drop_depth_prob=0.5,
        num_doa_bins=360,
        num_distance_bins=120,
        audio_in_channels=2,
        freeze_depth_encoder=False,
        dropout=0.1,
    ):
        super().__init__()

        self.drop_depth_prob = drop_depth_prob

        self.spec_encoder = SpecEncoderGlobal(
            in_channels=audio_in_channels,
            channels=(16, 32, 64),
            dropout=dropout,
            out_dim=spec_out_dim,
            use_compress=use_compress,
        )

        self.depth_encoder = DepthResNet18Encoder(
            out_dim=depth_out_dim,
            pretrained=pretrained_depth_encoder,
        )

        if freeze_depth_encoder and hasattr(self.depth_encoder, "features"):
            for param in self.depth_encoder.features.parameters():
                param.requires_grad = False

        self.film_gamma = nn.Linear(depth_out_dim, spec_out_dim)
        self.film_beta = nn.Linear(depth_out_dim, spec_out_dim)

        self.fusion_fc = nn.Sequential(
            nn.Linear(spec_out_dim, fusion_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.out_dim = fusion_out_dim

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

    def forward(self, spectrogram, depth):
        spec_feat = self.spec_encoder(spectrogram)
        depth_feat = self.depth_encoder(depth)

        if self.training and self.drop_depth_prob > 0.0:
            drop_mask = torch.rand(depth_feat.size(0), 1, device=depth_feat.device) < self.drop_depth_prob
            depth_feat = torch.where(drop_mask, torch.zeros_like(depth_feat), depth_feat)

        gamma = self.film_gamma(depth_feat)
        beta = self.film_beta(depth_feat)

        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)

        spec_film = gamma * spec_feat + beta
        out_feat = self.fusion_fc(spec_film)

        doa_logits = self.doa_head(out_feat)
        distance_logits = self.distance_head(out_feat)
        return doa_logits, distance_logits


class AudioVisualDOANet(nn.Module):
    """Compatibility wrapper used by main_audio_visual_doa.py."""

    def __init__(
        self,
        audio_in_channels,
        model_type="audio_depth",
        audio_feat_dim=256,
        image_feat_dim=128,
        depth_feat_dim=64,
        fusion_dim=256,
        num_doa_bins=360,
        num_distance_bins=120,
        dropout=0.0,
        use_compress=True,
        pretrained_depth_encoder=False,
        freeze_depth_encoder=False,
        drop_depth_prob=0.1,
    ):
        super().__init__()
        self.model_type = model_type
        self.use_depth = model_type in ("audio_depth", "audio_image_depth")

        if model_type in ("audio_image", "audio_image_depth"):
            print(f"[WARN] {model_type} selected, but this SSLNet variant ignores RGB image input.")
        if model_type not in ("audio", "audio_depth", "audio_image", "audio_image_depth"):
            raise ValueError(f"Unknown model_type: {model_type}")

        if self.use_depth:
            self.net = SSLNet_depth_DOA(
                spec_out_dim=audio_feat_dim,
                depth_out_dim=depth_feat_dim,
                fusion_out_dim=fusion_dim,
                pretrained_depth_encoder=pretrained_depth_encoder,
                use_compress=use_compress,
                drop_depth_prob=drop_depth_prob,
                num_doa_bins=num_doa_bins,
                num_distance_bins=num_distance_bins,
                audio_in_channels=audio_in_channels,
                freeze_depth_encoder=freeze_depth_encoder,
                dropout=dropout,
            )
        else:
            self.net = SSLNet_DOA(
                spec_out_dim=audio_feat_dim,
                fusion_out_dim=fusion_dim,
                pretrained_depth_encoder=pretrained_depth_encoder,
                use_compress=use_compress,
                num_doa_bins=num_doa_bins,
                num_distance_bins=num_distance_bins,
                audio_in_channels=audio_in_channels,
                dropout=dropout,
            )

    def forward(self, spectrogram, image=None, depth=None):
        if self.use_depth:
            if depth is None:
                raise ValueError("depth is required for audio_depth/audio_image_depth models")
            return self.net(spectrogram, depth=depth)
        return self.net(spectrogram)


if __name__ == "__main__":
    model = AudioVisualDOANet(audio_in_channels=12, model_type="audio_depth", dropout=0.0)
    spectrogram = torch.randn(2, 12, 99, 101)
    depth = torch.randn(2, 1, 224, 224)
    doa, distance = model(spectrogram, depth=depth)
    print(doa.shape, distance.shape)
