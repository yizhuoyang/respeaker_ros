import numpy as np
import torch
import torch.nn as nn
import warnings
import time
import sys
from torch.nn import MultiheadAttention

# sys.path.append("../")
from network.audionet.util import SteeringVector, ModeVector_torch
import torch.nn.functional as F


class RxCrossAttention(nn.Module):
    def __init__(self, M, embed_dim=128, num_heads=4):
        super(RxCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_proj = nn.Linear(2 * M * M, embed_dim)
        self.key_proj = nn.Linear(2 * M * M, embed_dim)
        self.value_proj = nn.Linear(2 * M * M, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc_out = nn.Linear(embed_dim, 2 * M * M)

    def forward(self, rx_trad, rx_dl):
        B, _, M, _ = rx_trad.shape
        rx_trad = rx_trad.view(B, 257, 2, M, M).view(B, 257, 2 * M * M)
        rx_dl = rx_dl.view(B, 257, 2, M, M).view(B, 257, 2 * M * M)
        Q = self.query_proj(rx_trad)  # (B, 257, embed_dim)
        K = self.key_proj(rx_dl)      # (B, 257, embed_dim)
        V = self.value_proj(rx_dl)    # (B, 257, embed_dim)
        attn_output, _ = self.attn(Q, K, V)  # (B, 257, embed_dim)
        out = self.fc_out(attn_output)  # (B, 257, 2 * M * M)
        out = out.view(B, 257, 2, M, M).permute(0, 2, 1, 3, 4).contiguous().view(B, 2 * 257, M, M)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn
#
class Encoder(nn.Module):
    def __init__(self, input_channel=8,dropout_prob=0.3):  # Add dropout_prob to control
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)#16
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) #32
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 257, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(257)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout2d(p=dropout_prob)  # Dropout2D for feature maps
        self.spatial_attn = SpatialAttention()  # Optional

    def forward(self, x):
        skip1 = F.relu(self.bn1(self.conv1(x)))
        # skip1 = self.dropout(skip1)
        x = self.pool1(skip1)

        skip2 = F.relu(self.bn2(self.conv2(x)))
        # skip2 = self.dropout(skip2)
        x = self.pool2(skip2)

        skip3 = F.relu(self.bn3(self.conv3(x)))
        # skip3 = self.dropout(skip3)
        x = self.pool3(skip3)

        skip4 = F.relu(self.bn4(self.conv4(x)))
        # skip4 = self.dropout(skip4)
        x = self.pool4(skip4)

        return x, [skip1, skip2, skip3, skip4]

class Encoder_cls(nn.Module):
    def __init__(self, dropout_prob=0.1,input_channel=8):  # Add dropout_prob to control
        super(Encoder_cls, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, padding=1)#16
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1) #32
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout2d(p=dropout_prob)  # Dropout2D for feature maps

    def forward(self, x):
        skip1 = F.relu(self.bn1(self.conv1(x)))
        # skip1 = self.dropout(skip1)
        x = self.pool1(skip1)

        skip2 = F.relu(self.bn2(self.conv2(x)))
        # skip2 = self.dropout(skip2)
        x = self.pool2(skip2)

        skip3 = F.relu(self.bn3(self.conv3(x)))
        # skip3 = self.dropout(skip3)
        x = self.pool3(skip3)

        skip4 = F.relu(self.bn4(self.conv4(x)))
        # skip4 = self.dropout(skip4)
        x = self.pool4(skip4)

        return x



# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)  # Batch Norm
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)  # Batch Norm
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)  # Batch Norm
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv4 = nn.Conv2d(64, 257, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(257)  # Batch Norm
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.spatial_attn = SpatialAttention()  # (Optional)
#
#     def forward(self, x):
#         skip1 = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool1(skip1)
#
#         skip2 = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool2(skip2)
#
#         skip3 = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool3(skip3)
#
#         skip4 = F.relu(self.bn4(self.conv4(x)))
#         x = self.pool4(skip4)
#         return x, [skip1, skip2, skip3, skip4]


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(8, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv4 = nn.Conv2d(64, 257, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(257)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.spatial_attn1 = SpatialAttention()
#         self.spatial_attn2 = SpatialAttention()
#         self.spatial_attn3 = SpatialAttention()
#         self.spatial_attn4 = SpatialAttention()
#
#     def forward(self, x):
#         skip1 = F.relu(self.bn1(self.conv1(x)))
#         skip1 = self.spatial_attn1(skip1)
#         x = self.pool1(skip1)
#
#         skip2 = F.relu(self.bn2(self.conv2(x)))
#         skip2 = self.spatial_attn2(skip2)
#         x = self.pool2(skip2)
#
#         skip3 = F.relu(self.bn3(self.conv3(x)))
#         skip3 = self.spatial_attn3(skip3)
#         x = self.pool3(skip3)
#
#         skip4 = F.relu(self.bn4(self.conv4(x)))
#         skip4 = self.spatial_attn4(skip4)
#         x = self.pool4(skip4)
#
#         return x, [skip1, skip2, skip3, skip4]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(257, 128, kernel_size=2, stride=2)
        self.bn_up1 = nn.BatchNorm2d(128)  # Batch Norm
        self.conv1 = nn.Conv2d(128 + 257, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)  # Batch Norm

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn_up2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.bn_up3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.bn_up4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.final_conv = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, x, skips):
        x = self.bn_up1(self.up1(x))
        x = torch.cat([x, skips[3]], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.bn_up2(self.up2(x))
        x = torch.cat([x, skips[2]], dim=1)
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.bn_up3(self.up3(x))
        x = torch.cat([x, skips[1]], dim=1)
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.bn_up4(self.up4(x))
        x = F.interpolate(x, size=skips[0].shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skips[0]], dim=1)
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.final_conv(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.DropOut = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu =  nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 257*2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(257*2)

        self.res1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.res2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.res3 = nn.Conv2d(256, 257*2, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=(4,2))

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.DropOut(x1)
        x1_p = self.pool(x1)
        x2 = self.relu(self.bn2(self.conv2(x1_p)))
        x2 = self.DropOut(x2)
        x2_res = self.res1(x1_p)
        x2 = x2 + x2_res
        x2_p = self.pool(x2)

        x3 = self.relu(self.bn3(self.conv3(x2_p)))
        x3 = self.DropOut(x3)
        x3_res = self.res2(x2_p)
        x3 = x3 + x3_res
        x3_p = self.pool2(x3)

        x4 = self.relu(self.bn4(self.conv4(x3_p)))
        x4 = self.DropOut(x4)
        x4_res = self.res3(x3_p)
        x4 = x4 + x4_res
        x4_p = self.pool2(x4)
        return x4_p


class Grid:
    def __init__(self):
        self.x = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_x.npy')
        self.y = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_y.npy')
        self.z = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_z.npy')
        
def soft_argmax_peak_refined(spectrum, alpha=20.0, top_k=5):
    batch_size, num_angles = spectrum.shape
    angles = torch.linspace(0, 360, spectrum.shape[-1], device=spectrum.device)
    top_indices = torch.topk(spectrum, k=top_k, dim=-1).indices
    top_spectrums = torch.gather(spectrum, dim=-1, index=top_indices)
    top_angles = torch.gather(angles.expand(batch_size, -1), dim=-1, index=top_indices)
    weights = torch.softmax(alpha * top_spectrums, dim=-1)
    doa_estimation = torch.sum(weights * top_angles, dim=-1)
    return doa_estimation


class DeepMusic_pretrain(nn.Module):
    def __init__(self,input_channel=8):
        super(DeepMusic_pretrain, self).__init__()
        self.encoder = Encoder(input_channel=input_channel)
        self.decoder = Decoder()
    def forward(self, x: torch.Tensor):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DeepMusic_plus(nn.Module):
    def __init__(self, N, T, M,device,attention=True,input_channel=8):
        super(DeepMusic_plus, self).__init__()
        self.N, self.T, self.M = N, T, M
        self.input_channel = input_channel
        self.angels = torch.linspace(-1 * np.pi , np.pi, 360).to(device)
        self.device = device
        self.input_size = self.N
        self.hidden_size = 2 * self.N
        self.DropOut = nn.Dropout(0.1)
        self.autoencoder = Autoencoder()
        self.encoder = Encoder(input_channel)
        self.conv2d1 = nn.Conv2d(in_channels=257, out_channels=257, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2d2 = nn.Conv2d(in_channels=257, out_channels=257, kernel_size=3, stride=1, padding=1, bias=False)
        self.convs2  = nn.Conv1d(in_channels=257, out_channels=257, kernel_size=3, stride=1, padding=1)  # Output: (16, 32, 32)
        self.convs1  = nn.Conv1d(in_channels=257, out_channels=1, kernel_size=3, stride=1, padding=1)  # Output: (16, 32, 32)
        self.relu    =  nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.resnet = resnet()
        self.fc = nn.Linear(64,int(input_channel*input_channel/2))
        self.atten1 = RxCrossAttention(self.N)
        self.atten2 = RxCrossAttention(self.N)
        self.dropout = nn.Dropout2d(p=0.3)
        self.channel_attention = ChannelAttention(in_channels=257, reduction=16)
        self.attention = attention
    def spectrum_calculation(self, Un,sv):
        batch_size, F, M, _ = Un.shape
        Un_UnH = torch.matmul(Un, torch.conj(Un).transpose(-2, -1))
        sv = sv.to(dtype=Un.dtype, device=Un.device)
        bs,_, num_angles, _ = sv.shape
        # sv = sv.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, F, N_angles, M]
        sv_H = torch.conj(sv).transpose(-2, -1)  # [batch_size, F, M, N_angles]
        denominator_matrix = torch.matmul(sv_H, torch.matmul(Un_UnH, sv))
        spectrum_eq = denominator_matrix.diagonal(dim1=-2, dim2=-1).real
        spectrum_eq = spectrum_eq.clamp(min=1e-6)
        spectrum = 1 / spectrum_eq
        return spectrum

    def pre_MUSIC(self,Rz,sv):
        # Rz = Rz + 1e-5 * torch.eye(Rz.shape[-1], device=Rz.device)
        Rz = (Rz + Rz.transpose(-1, -2).conj()) / 2
        # TODO modification
        # Rz = Rz @ Rz.transpose(-1, -2).conj()  

        Rz = Rz + 1e-5 * torch.eye(Rz.shape[-1], device=Rz.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(Rz)
        eigenvectors = torch.flip(eigenvectors, dims=[-1])
        Un = eigenvectors[:, :, :, self.M:]
        return self.spectrum_calculation(Un,sv)

    def forward(self, X,sv,correlation):
        self.BATCH_SIZE = X.shape[0]
        denoised_spec = X
        x,_ = self.encoder(denoised_spec)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.fc(x)
        x = x.view(x.shape[0], x.shape[1], int(self.input_channel/2), self.input_channel)
        Rx_real, Rx_imag = torch.chunk(x, chunks=2, dim=-1)
        feature = torch.complex(Rx_real, Rx_imag)
        # correlation_real, correlation_imgae = correlation.real,correlation.imag
        # feature_rx = torch.concatenate([Rx_real,Rx_imag],1)
        # #
        # feature_correlation = torch.concatenate([correlation_real,correlation_imgae],1)
        # feature1 = self.atten1(feature_rx,feature_correlation)
        # feature  = feature1
        # # feature2 = self.atten2(feature_rx,feature_correlation)
        # # feature  = feature1+feature2
        # Rx_real, Rx_imag = torch.chunk(feature, chunks=2, dim=1)
        # feature = torch.complex(Rx_real, Rx_imag)
        # print(feature[0][100],correlation[0][100])
        # feature = (correlation+feature)/2
        # print(feature)
        # feature = self.relu(self.conv2d1(feature))
        # feature = self.conv2d2(feature)
        spectrum  = self.pre_MUSIC(feature,sv)
        # spectrum2 = self.pre_MUSIC(correlation,sv)
        # spectrum  = torch.cat([spectrum,spectrum2],dim=1)
        spectrum = self.relu(self.convs2(spectrum))
        #use channel attention
        if self.attention:
            spectrum = self.channel_attention(spectrum)
        # spectrum = self.dropout(spectrum)
        spectrum = self.sigmoid(self.convs1(spectrum))
        # spectrum = self.dropout(spectrum)
        spectrum = spectrum.squeeze(1)
        DOA = soft_argmax_peak_refined(spectrum).unsqueeze(1)
        return DOA,spectrum

class DeepMusic_plus_class(nn.Module):
    def __init__(self, N, T, M,device,input_channel=8):
        super(DeepMusic_plus_class, self).__init__()
        self.N, self.T, self.M = N, T, None
        self.input_channel = input_channel
        self.angels = torch.linspace(-1 * np.pi , np.pi, 360).to(device)
        self.device = device
        self.input_size = self.N
        self.hidden_size = 2 * self.N
        self.DropOut = nn.Dropout(0.1)
        self.encoder = Encoder(input_channel=input_channel)
        self.encoder_class = Encoder_cls(input_channel=input_channel)
        self.convs2  = nn.Conv1d(in_channels=257, out_channels=257, kernel_size=3, stride=1, padding=1)  # Output: (16, 32, 32)
        self.convs1  = nn.Conv1d(in_channels=257, out_channels=1, kernel_size=3, stride=1, padding=1)  # Output: (16, 32, 32)
        self.relu    =  nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        # self.resnet = resnet()
        self.fc = nn.Linear(64,int(input_channel*input_channel/2))
        self.dropout = nn.Dropout2d(p=0.3)
        self.channel_attention = ChannelAttention(in_channels=257, reduction=16)
        self.source_prediction1 = nn.Linear(64*32,128)
        self.source_prediction2 = nn.Linear(128,self.N)

    def spectrum_calculation(self, Un_list, sv):
        spectra = []
        for b, Un in enumerate(Un_list):
            Un_UnH = torch.matmul(Un, torch.conj(Un).transpose(-2, -1))
            sv_b = sv[b].to(dtype=Un.dtype, device=Un.device)
            sv_H = torch.conj(sv_b).transpose(-2, -1)  # [batch_size, F, M, N_angles]
            denominator_matrix = torch.matmul(sv_H, torch.matmul(Un_UnH, sv_b))
            spectrum_eq = denominator_matrix.diagonal(dim1=-2, dim2=-1).real
            spectrum_eq = spectrum_eq.clamp(min=1e-6)
            spectrum = 1 / spectrum_eq
            spectra.append(spectrum)
        return torch.stack(spectra, dim=0)


    def pre_MUSIC(self,Rz,sv):
        Rz = (Rz + Rz.transpose(-1, -2).conj()) / 2
        Rz = Rz + 1e-5 * torch.eye(Rz.shape[-1], device=Rz.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(Rz)
        eigenvectors = torch.flip(eigenvectors, dims=[-1])
        Un_list = []
        for b in range(Rz.shape[0]):
            M_b = self.M[b].item()
            Un_b = eigenvectors[b, :,:, M_b:]  # shape: [C, C-M]
            Un_list.append(Un_b)
        return self.spectrum_calculation(Un_list, sv)


    def forward(self, X,sv,correlation):
        self.BATCH_SIZE = X.shape[0]
        denoised_spec = X
        num_source = self.encoder_class(denoised_spec)
        num_source = num_source.view(num_source.shape[0],-1)
        num_source = self.relu(self.source_prediction1(num_source))
        num_source = self.source_prediction2(num_source)
        self.M     = torch.argmax(num_source, dim=1)

        x,_ = self.encoder(denoised_spec)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.fc(x)
        x = x.view(x.shape[0], x.shape[1], int(self.input_channel/2), self.input_channel)
        # x = x.view(x.shape[0], x.shape[1], 4, 8)
        Rx_real, Rx_imag = torch.chunk(x, chunks=2, dim=-1)
        feature = torch.complex(Rx_real, Rx_imag)
        spectrum  = self.pre_MUSIC(feature,sv)
        spectrum = self.relu(self.convs2(spectrum))
        spectrum = self.channel_attention(spectrum)
        spectrum = self.sigmoid(self.convs1(spectrum))
        spectrum = spectrum.squeeze(1)
        DOA = soft_argmax_peak_refined(spectrum).unsqueeze(1)
        return DOA,spectrum,num_source




class DAMUSIC_class(nn.Module):
    def __init__(self, N, T, M,device,input_channel=8):
        super(DAMUSIC_class, self).__init__()
        self.N, self.T, self.M = N, T, None
        self.input_channel = input_channel
        self.angels = torch.linspace(-1 * np.pi , np.pi, 360).to(device)
        self.device = device
        self.input_size = self.N
        self.hidden_size = 2 * self.N
        self.DropOut = nn.Dropout(0.1)
        self.encoder = Encoder(input_channel=input_channel)
        self.encoder_class = Encoder_cls(input_channel=input_channel)
        self.convs2  = nn.Conv1d(in_channels=257, out_channels=257, kernel_size=3, stride=1, padding=1)  # Output: (16, 32, 32)
        self.convs1  = nn.Conv1d(in_channels=257, out_channels=1, kernel_size=3, stride=1, padding=1)  # Output: (16, 32, 32)
        self.fc1 = nn.Linear(self.angels.shape[0]*257, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3) 
        
        self.relu    =  nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.resnet = resnet()
        self.fc = nn.Linear(64,int(input_channel*input_channel/2))
        self.dropout = nn.Dropout2d(p=0.3)
        self.channel_attention = ChannelAttention(in_channels=257, reduction=16)
        self.source_prediction1 = nn.Linear(64*32,128)
        self.source_prediction2 = nn.Linear(128,3)

    def spectrum_calculation(self, Un_list, sv):
        spectra = []
        for b, Un in enumerate(Un_list):
            Un_UnH = torch.matmul(Un, torch.conj(Un).transpose(-2, -1))
            sv_b = sv[b].to(dtype=Un.dtype, device=Un.device)
            sv_H = torch.conj(sv_b).transpose(-2, -1)  # [batch_size, F, M, N_angles]
            denominator_matrix = torch.matmul(sv_H, torch.matmul(Un_UnH, sv_b))
            spectrum_eq = denominator_matrix.diagonal(dim1=-2, dim2=-1).real
            spectrum_eq = spectrum_eq.clamp(min=1e-6)
            spectrum = 1 / spectrum_eq
            spectra.append(spectrum)
        return torch.stack(spectra, dim=0)

    def pre_MUSIC(self,Rz,sv):
        Rz = (Rz + Rz.transpose(-1, -2).conj()) / 2
        Rz = Rz + 1e-5 * torch.eye(Rz.shape[-1], device=Rz.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(Rz)
        eigenvectors = torch.flip(eigenvectors, dims=[-1])
        Un_list = []
        for b in range(Rz.shape[0]):
            M_b = self.M[b].item()
            Un_b = eigenvectors[b, :,:, M_b:]  # shape: [C, C-M]
            Un_list.append(Un_b)
        return self.spectrum_calculation(Un_list, sv)


    def forward(self, X,sv,correlation):
        self.BATCH_SIZE = X.shape[0]
        denoised_spec = X
        num_source = self.encoder_class(denoised_spec)
        num_source = num_source.view(num_source.shape[0],-1)
        num_source = self.relu(self.source_prediction1(num_source))
        num_source = self.source_prediction2(num_source)
        self.M     = torch.argmax(num_source, dim=1)

        x,_ = self.encoder(denoised_spec)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.fc(x)
        x = x.view(x.shape[0], x.shape[1], int(self.input_channel/2), self.input_channel)
        # x = x.view(x.shape[0], x.shape[1], 4, 8)
        Rx_real, Rx_imag = torch.chunk(x, chunks=2, dim=-1)
        feature = torch.complex(Rx_real, Rx_imag)
        spectrum  = self.pre_MUSIC(feature,sv)
        spectrum = spectrum.view(self.BATCH_SIZE,-1)
        DOA = self.relu(self.fc1(spectrum))
        DOA = self.relu((self.fc2(DOA)))
        DOA = self.fc3(DOA)
        return DOA,num_source


class Music(nn.Module):
    def __init__(self, N, T, M, mic_positions,device):
        super(Music, self).__init__()
        self.N, self.T, self.M = N, T, M
        self.angels = torch.linspace(-1 * np.pi , np.pi, 360).to(device)
        self.input_size = self.N
        grid = Grid()
        steer_vector_calc = ModeVector_torch(torch.tensor(mic_positions.T), 16000, 512, 343, grid, "far", precompute=True, device='cuda')
        self.sv = steer_vector_calc.mode_vec

    def spectrum_calculation(self, Un: torch.Tensor):
        batch_size, F, M, _ = Un.shape
        Un_UnH = torch.matmul(Un, torch.conj(Un).transpose(-2, -1))
        sv = self.sv.to(dtype=Un.dtype, device=Un.device)
        _, num_angles, _ = sv.shape
        sv = sv.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, F, N_angles, M]
        sv_H = torch.conj(sv).transpose(-2, -1)  # [batch_size, F, M, N_angles]
        denominator_matrix = torch.matmul(sv_H, torch.matmul(Un_UnH, sv))
        spectrum_eq = denominator_matrix.diagonal(dim1=-2, dim2=-1).real
        spectrum_eq = spectrum_eq.clamp(min=1e-6)
        spectrum = 1 / spectrum_eq
        return spectrum

    def pre_MUSIC(self,Rz,):
        eigenvalues, eigenvectors = torch.linalg.eigh(Rz)
        eigenvectors = torch.flip(eigenvectors, dims=[-1])
        Un = eigenvectors[:, :, :, self.M:]
        return self.spectrum_calculation(Un)

    def forward(self, X: torch.Tensor):
        self.BATCH_SIZE = X.shape[0]
        spectrum = self.pre_MUSIC(X)
        return spectrum
