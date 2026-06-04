import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def onnx_safe_adaptive_avg_pool2d(x, output_size):
    if not torch.onnx.is_in_onnx_export():
        return F.adaptive_avg_pool2d(x, output_size)

    out_h, out_w = output_size
    in_h, in_w = int(x.shape[-2]), int(x.shape[-1])
    if in_h == out_h and in_w == out_w:
        return x
    if in_h % out_h == 0 and in_w % out_w == 0:
        kernel = (in_h // out_h, in_w // out_w)
        return F.avg_pool2d(x, kernel_size=kernel, stride=kernel)
    if in_h == out_h and in_w == 1 and out_w > 1:
        return x.expand(-1, -1, out_h, out_w)
    return F.interpolate(x, size=output_size, mode="area")


class DepthResNet18Encoder(nn.Module):
    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()

        try:
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)

        old_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None,
        )

        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.out_dim = out_dim
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        """
        x: (B, 1, 128, 128)
        """
        x = self.features(x)              # (B, 512, H', W')
        x = self.global_pool(x)           # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)         # (B, 512)
        x = self.fc(x)                    # (B, out_dim)
        return x

class HeatmapDecoder(nn.Module):

    def __init__(self, in_dim=256):
        super().__init__()

        # 16x16 -> 32x32
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # (B, 1, 32, 32)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # (B, 32, 64, 64)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # (B, 32, 64, 64)
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )


        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (B, 256)
        """
        B = x.size(0)
        x = x.view(B, 1, 16, 16)

        x = self.up1(x)        # (B, 32, 32, 32)
        x = self.up2(x)        # (B, 16, 64, 64)
        # x = self.up3(x)
        x = self.out_conv(x)   # (B, 1, 64, 64)
        # x = torch.sigmoid(x)
        # x = x.squeeze(1)  
        return x

class SpecEncoderGlobal(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels=(16, 32, 64),
        dropout: float = 0.1,
        out_dim: int = 256,
        use_compress: bool = True,
    ):
        super().__init__()
        self.use_compress = use_compress
        self.out_dim = out_dim
        if use_compress:
            conv_channels = channels
        else:
            conv_channels = (16, 32, 64, 64, 64)

        conv_blocks = []
        prev_c = in_channels

        for ch in conv_channels:
            block = nn.Sequential(
                nn.Conv2d(prev_c, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # H,W 都 /2
                nn.Dropout2d(dropout),
            )
            conv_blocks.append(block)
            prev_c = ch

        self.conv = nn.Sequential(*conv_blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((8, 3))
        flattened_dim = prev_c * 8 * 3
        self.fc = nn.Linear(flattened_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B = x.size(0)
        x = self.conv(x)
        x = onnx_safe_adaptive_avg_pool2d(x, (8, 3))   # -> (B, C_last, 8, 3)
        x = x.reshape(B, -1)      # -> (B, C_last*8*3)
        x = self.fc(x)            # -> (B, out_dim)
        return x


class EgoMapEncoder(nn.Module):
    def __init__(self, in_channels=2, base_channels=32, out_dim=128,
                 return_spatial=False):
        super().__init__()
        self.return_spatial = return_spatial

        # 31x31 -> 31x31
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        # 31x31 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        # 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        # 8x8 -> 4x4
        self.block4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.fc = nn.Linear(base_channels * 4, out_dim)

        self.out_dim = out_dim

    def forward(self, x):
        """
        x: (B, 2, 31, 31)
        """
        x = self.block1(x) 
        x = self.block2(x)  
        x = self.block3(x)  
        x = self.block4(x)  

        if self.return_spatial:
            return x

        x = self.global_pool(x)         
        x = x.view(x.size(0), -1)       
        x = self.fc(x)                   
        return x


class RgbResNet18Encoder(nn.Module):
    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()

        # 加载 ResNet18 backbone
        try:
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)


        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.out_dim = out_dim
        self.fc = nn.Linear(512, out_dim)  
        
    def forward(self, x):
        """
        x: (B, 3, 128, 128)
        """
        x = self.features(x)              # (B, 512, H', W')
        x = self.global_pool(x)           # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)         # (B, 512)
        x = self.fc(x)                    # (B, out_dim)
        return x


def gaussian_render_batch(
    coords: torch.Tensor,
    map_size: int = 64,
    meters_per_pixel: float = 0.5,
    base_sigma: float = 0.2,
    sigma_scale: float = 0.5,
):

    device = coords.device
    dtype = coords.dtype

    B = coords.size(0)
    H = W = map_size

    cx = W // 2
    cy = H // 2

    front = coords[:, 0]  # (B,)
    right = coords[:, 1]  # (B,)

    dx_pix = right / meters_per_pixel               # (B,)
    dy_pix = -front / meters_per_pixel              # (B,) front>0 向上

    source_x = cx + dx_pix                          # (B,)
    source_y = cy + dy_pix                          # (B,)

    dist_m = torch.sqrt(front**2 + right**2)        # (B,)
    sigma_m = base_sigma + sigma_scale * dist_m     # (B,)
    sigma_m = torch.clamp(sigma_m, min=1e-3)


    sigma = sigma_m / meters_per_pixel              # (B,)


    ys = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)  # (1, H, 1)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)  # (1, 1, W)

    source_x = source_x.view(B, 1, 1)
    source_y = source_y.view(B, 1, 1)
    sigma = sigma.view(B, 1, 1)

    gauss = torch.exp(
        -((xs - source_x) ** 2 + (ys - source_y) ** 2) / (2 * sigma ** 2)
    )  

    gauss = gauss.view(B, -1)                        # (B, H*W)
    sums = gauss.sum(dim=1, keepdim=True)           # (B, 1)
    gauss = gauss / (sums + 1e-8)                   # 防止除 0
    gauss = gauss.view(B, 1, H, W)                  # (B, 1, H, W)
    g_flat = gauss.view(B, -1)                      # (B, H*W)
    g_min = g_flat.min(dim=1, keepdim=True)[0]
    g_max = g_flat.max(dim=1, keepdim=True)[0]
    gauss = (g_flat - g_min) / (g_max - g_min + 1e-8)
    gauss = gauss.view(B, 1, H, W)

    return gauss

if __name__ == "__main__":
    # x = torch.randn(8, 2, 65, 26)
    # encoder = SpecEncoderGlobal(out_dim=256)
    # feat = encoder(x)
    # print(feat.shape)

    # depth = torch.randn(8, 1, 128, 128)
    # encoder = DepthResNet18Encoder(out_dim=256, pretrained=True)
    # feat = encoder(depth)
    # print("feature shape:", feat.shape)   # -> (4, 256)


    spec_enc_small = SpecEncoderGlobal(
        in_channels=2,
        channels=(16, 32, 64),
        dropout=0.1,
        out_dim=256,
        use_compress=True,
    )

    x_small = torch.randn(4, 2, 65, 26)
    feat_small = spec_enc_small(x_small)
    print(feat_small.shape)  

    spec_enc_big = SpecEncoderGlobal(
        in_channels=2,
        dropout=0.1,
        out_dim=256,
        use_compress=False, 
    )

    x_big = torch.randn(4, 2, 257, 101)
    feat_big = spec_enc_big(x_big)
    print(feat_big.shape) 
