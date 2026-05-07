import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
from network.audionet.faeture_extraction import SpecEncoderGlobal, DepthResNet18Encoder,HeatmapDecoder,gaussian_render_batch,RgbResNet18Encoder,EgoMapEncoder
# from faeture_extraction import SpecEncoderGlobal, DepthResNet18Encoder,HeatmapDecoder,gaussian_render_batch,RgbResNet18Encoder,EgoMapEncoder

class SSLNet(nn.Module):
    def __init__(
        self,
        spec_out_dim=256,
        depth_out_dim=64,
        fusion_out_dim=256,
        pretrained_depth_encoder=True,
        use_compress=True,
    ):
        super().__init__()

        self.spec_encoder = SpecEncoderGlobal(
            in_channels=2,
            channels=(16, 32, 64),
            dropout=0.1,
            out_dim=spec_out_dim,
            use_compress=use_compress
        )

        self.depth_encoder = DepthResNet18Encoder(
            out_dim=depth_out_dim,
            pretrained=pretrained_depth_encoder,
        )

        for p in self.depth_encoder.features.parameters():
            p.requires_grad = False

        self.fusion_fc = nn.Sequential(
            nn.Linear(spec_out_dim+depth_out_dim, fusion_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.out_dim = fusion_out_dim
        self.decoder = HeatmapDecoder()

        self.film_gamma = nn.Linear(depth_out_dim, spec_out_dim)
        self.film_beta  = nn.Linear(depth_out_dim, spec_out_dim)

    def forward(self, spectrogram, depth):
        spec_feat = self.spec_encoder(spectrogram)  # (B, spec_out_dim)
        heatmap = self.decoder(spec_feat)   
        return heatmap


class SSLNet_depth(nn.Module):
    def __init__(
        self,
        spec_out_dim=256,
        depth_out_dim=64,
        fusion_out_dim=256,
        pretrained_depth_encoder=True,
        use_compress=True,
        drop_depth_prob=0.5,  
    ):
        super().__init__()

        self.drop_depth_prob = drop_depth_prob

        # ===== Audio encoder =====
        self.spec_encoder = SpecEncoderGlobal(
            in_channels=2,
            channels=(16, 32, 64),
            dropout=0.1,
            out_dim=spec_out_dim,
            use_compress=use_compress,
        )

        # ===== Depth encoder =====
        self.depth_encoder = DepthResNet18Encoder(
            out_dim=depth_out_dim,
            pretrained=pretrained_depth_encoder,
        )


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
        self.decoder = HeatmapDecoder()

    def forward(self, spectrogram, depth):
        """
        spectrogram: (B, 2, 65/257, 26/101)
        depth:       (B, 3, H, W)
        return:      heatmap (B, 1, 64, 64)
        """
        B = spectrogram.size(0)

        # ===== 1. Audio feature =====
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
        heatmap_logits = self.decoder(out_feat)        # (B, 1, 64, 64)

        return heatmap_logits


class SSLNet_rgb(nn.Module):
    def __init__(
        self,
        spec_out_dim=256,
        depth_out_dim=64,
        fusion_out_dim=256,
        pretrained_depth_encoder=True,
        use_compress=True,
        drop_depth_prob=0.5,  
    ):
        super().__init__()

        self.drop_depth_prob = drop_depth_prob

        # ===== Audio encoder =====
        self.spec_encoder = SpecEncoderGlobal(
            in_channels=2,
            channels=(16, 32, 64),
            dropout=0.1,
            out_dim=spec_out_dim,
            use_compress=use_compress,
        )

        # ===== Depth encoder =====
        self.depth_encoder = RgbResNet18Encoder(
            out_dim=depth_out_dim,
            pretrained=pretrained_depth_encoder,
        )


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
        self.decoder = HeatmapDecoder()

    def forward(self, spectrogram, depth):
        """
        spectrogram: (B, 2, 65/257, 26/101)
        depth:       (B, 3, H, W)
        return:      heatmap (B, 1, 64, 64)
        """
        B = spectrogram.size(0)

        # ===== 1. Audio feature =====
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
        heatmap_logits = self.decoder(out_feat)        # (B, 1, 64, 64)

        return heatmap_logits

class SSLNet_ego(nn.Module):
    def __init__(
        self,
        spec_out_dim=256,
        depth_out_dim=64,
        fusion_out_dim=256,
        pretrained_depth_encoder=True,
        use_compress=True,
        drop_depth_prob=0.5,  
    ):
        super().__init__()

        self.drop_depth_prob = drop_depth_prob

        # ===== Audio encoder =====
        self.spec_encoder = SpecEncoderGlobal(
            in_channels=2,
            channels=(16, 32, 64),
            dropout=0.1,
            out_dim=spec_out_dim,
            use_compress=use_compress,
        )

        # ===== Depth encoder =====
        self.depth_encoder = EgoMapEncoder(
            out_dim=depth_out_dim
        )


        self.film_gamma = nn.Linear(depth_out_dim, spec_out_dim)
        self.film_beta  = nn.Linear(depth_out_dim, spec_out_dim)

        self.fusion_fc = nn.Sequential(
            nn.Linear(spec_out_dim, fusion_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.out_dim = fusion_out_dim

        # Heatmap decoder: (B, fusion_out_dim) -> (B, 1, 64, 64)
        self.decoder = HeatmapDecoder()

    def forward(self, spectrogram, depth):
        """
        spectrogram: (B, 2, 65/257, 26/101)
        depth:       (B, 3, H, W)
        return:      heatmap (B, 1, 64, 64)
        """
        B = spectrogram.size(0)

        # ===== 1. Audio feature =====
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
        heatmap_logits = self.decoder(out_feat)        # (B, 1, 64, 64)

        return heatmap_logits


class SSLNet_DOA(nn.Module):
    def __init__(
        self,
        spec_out_dim=256,
        depth_out_dim=64,
        fusion_out_dim=256,
        pretrained_depth_encoder=True,
        use_compress=True,        
        num_doa_bins=360,   # 新增：DOA 高斯的维度
        num_distance_bins=120
    ):
        super().__init__()

        self.spec_encoder = SpecEncoderGlobal(
            in_channels=2,
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


    def forward(self, spectrogram, depth):
        spec_feat = self.spec_encoder(spectrogram)  
        
        doa_logits = self.doa_head(spec_feat)    
        doa_logits = torch.sigmoid(doa_logits)        
        
        distance_logits = self.distance_head(spec_feat)
        distance_logits = torch.sigmoid(distance_logits)  

        return doa_logits,distance_logits

class SSLNet_depth_DOA(nn.Module):
    def __init__(
        self,
        spec_out_dim=256,
        depth_out_dim=64,
        fusion_out_dim=256,
        pretrained_depth_encoder=True,
        use_compress=True,
        drop_depth_prob=0.5,  
        num_doa_bins=360,   # 新增：DOA 高斯的维度
        num_distance_bins=120,
    ):
        super().__init__()

        self.drop_depth_prob = drop_depth_prob

        # ===== Audio encoder =====
        self.spec_encoder = SpecEncoderGlobal(
            in_channels=2,
            channels=(16, 32, 64),
            dropout=0.1,
            out_dim=spec_out_dim,
            use_compress=use_compress,
        )

        # ===== Depth encoder =====
        self.depth_encoder = DepthResNet18Encoder(
            out_dim=depth_out_dim,
            pretrained=pretrained_depth_encoder,
        )


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
        doa_logits = torch.sigmoid(doa_logits)         
        
        # Distance head
        distance_logits = self.distance_head(out_feat)
        distance_logits = torch.sigmoid(distance_logits)        
        return doa_logits,distance_logits




class SSLNet_depth_DOA_res(nn.Module):
    def __init__(
        self,
        spec_out_dim=512,
        depth_out_dim=64,
        fusion_out_dim=512,
        pretrained_depth_encoder=True,
        use_compress=True,
        drop_depth_prob=0.5,  
        num_doa_bins=360,   # 新增：DOA 高斯的维度
        num_distance_bins=120,
    ):
        super().__init__()

        self.drop_depth_prob = drop_depth_prob

        # ===== Audio encoder =====
        self.spec_encoder = models.resnet18(pretrained=True)
        self.spec_encoder.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.spec_encoder.fc = nn.Identity()
        self.doa_head = nn.Linear(512,360)
        self.dis_head = nn.Linear(512,120)

        # ===== Depth encoder =====
        self.depth_encoder = DepthResNet18Encoder(
            out_dim=depth_out_dim,
            pretrained=pretrained_depth_encoder,
        )


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

        # # Heatmap decoder: (B, fusion_out_dim) -> (B, 1, 64, 64)
        # self.num_doa_bins = num_doa_bins
        # self.doa_head = nn.Sequential(
        #     nn.Linear(spec_out_dim, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, num_doa_bins),
        # )

        # self.num_distance_bins = num_distance_bins
        # self.distance_head = nn.Sequential(
        #     nn.Linear(spec_out_dim, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, num_distance_bins),
        # )

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
        doa_logits = torch.sigmoid(doa_logits)         
        
        # Distance head
        distance_logits = self.dis_head(out_feat)
        distance_logits = torch.sigmoid(distance_logits)        
        return doa_logits,distance_logits



class SSLNet_DOA_res(nn.Module):
    def __init__(
        self,
        spec_out_dim=512,
        depth_out_dim=64,
        fusion_out_dim=512,
        pretrained_depth_encoder=True,
        use_compress=True,
        drop_depth_prob=0.5,  
    ):
        super().__init__()

        self.drop_depth_prob = drop_depth_prob

        # ===== Audio encoder =====
        self.spec_encoder = models.resnet18(pretrained=True)
        self.spec_encoder.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.spec_encoder.fc = nn.Identity()
        self.doa_head = nn.Linear(512,360)
        self.dis_head = nn.Linear(512,120)

        # ===== Depth encoder =====
        self.depth_encoder = DepthResNet18Encoder(
            out_dim=depth_out_dim,
            pretrained=pretrained_depth_encoder,
        )


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

    def forward(self, spectrogram, depth):
        """
        spectrogram: (B, 2, 65/257, 26/101)
        depth:       (B, 3, H, W)
        return:      heatmap (B, 1, 64, 64)
        """
        B = spectrogram.size(0)
        spec_feat = self.spec_encoder(spectrogram)     # (B, spec_out_dim)

        out_feat = self.fusion_fc(spec_feat)           # (B, fusion_out_dim)
        
        # DOA head
        doa_logits = self.doa_head(out_feat)
        doa_logits = torch.sigmoid(doa_logits)         
        
        # Distance head
        distance_logits = self.dis_head(out_feat)
        distance_logits = torch.sigmoid(distance_logits)        
        return doa_logits,distance_logits



class SSLNetMultiFrame(nn.Module):
    def __init__(
        self,
        spec_out_dim=256,
        fusion_out_dim=256,    # 最终给 decoder 的 dim
        pose_emb_dim=32,
        use_compress=True,
    ):
        super().__init__()

        if fusion_out_dim is None:
            fusion_out_dim = spec_out_dim


        self.spec_encoder = SpecEncoderGlobal(
            in_channels=2,
            channels=(16, 32, 64),
            dropout=0.1,
            out_dim=spec_out_dim,
            use_compress=use_compress,
        )

        self.pose_mlp = nn.Sequential(
            nn.Linear(4, pose_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pose_emb_dim, pose_emb_dim),
            nn.ReLU(inplace=True),
        )
        self.gru_hidden = fusion_out_dim // 2

        self.gru = nn.GRU(
            input_size=spec_out_dim + pose_emb_dim,
            hidden_size=self.gru_hidden,
            num_layers=1,
            batch_first=True,     
            bidirectional=True,  
        )

        self.out_dim = fusion_out_dim 
        # ========= 4. Heatmap Decoder =========
        self.decoder = HeatmapDecoder() 

    def forward(self, spectrogram_seq, rel_pose_seq):
        """
        Args:
            spectrogram_seq: [B, K, 2, F, T]
            rel_pose_seq:    [B, K, 4]

        Returns:
            heatmap: [B, 1, Hm, Wm]
        """
        B, K, C, Freq, Time = spectrogram_seq.shape

        spec_flat = spectrogram_seq.view(B * K, C, Freq, Time)
        spec_feat_flat = self.spec_encoder(spec_flat)        # [B*K, spec_out_dim]
        spec_feat_seq = spec_feat_flat.view(B, K, -1)        # [B, K, spec_out_dim]

        pose_emb = self.pose_mlp(rel_pose_seq)               # [B, K, pose_emb_dim]
        fusion_seq = torch.cat([spec_feat_seq, pose_emb], dim=-1)  # [B, K, spec_out_dim+pose_emb_dim]

        # h_seq: [B, K, 2*hidden] = [B, K, fusion_out_dim]
        # h_last: [num_layers*2, B, hidden]
        h_seq, h_last = self.gru(fusion_seq)
        global_feat = h_seq[:, -1, :]                         # [B, fusion_out_dim]
        heatmap = self.decoder(global_feat)                   # [B, 1, Hm, Wm]

        return heatmap


# if __name__ == "__main__":
#     audio = torch.randn(4, 2, 65, 26)
#     depth = torch.randn(4, 1, 128, 128)
#     model = SSLNet_depth()
#     out = model(audio, depth)
#     print(out.shape)  


if __name__ == "__main__":
    B = 4
    K = 3
    C = 2
    F = 65
    T = 26
    spectrogram_seq = torch.randn(B, K, C, F, T)  # [B,K,2,65,26]
    rel_pose_seq    = torch.randn(B, K, 4)        # [B,K,4], (front,right,sinθ,cosθ) 这里只是测试随便填


    model = SSLNetMultiFrame(
        spec_out_dim=256,
        fusion_out_dim=256,  
        pose_emb_dim=32,
        use_compress=True,
    )

    out = model(spectrogram_seq, rel_pose_seq)

    print("Input spectrogram_seq shape:", spectrogram_seq.shape)
    print("Input rel_pose_seq shape:",    rel_pose_seq.shape)
    print("Output heatmap shape:",        out.shape)