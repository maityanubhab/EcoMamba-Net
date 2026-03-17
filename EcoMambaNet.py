
"""
@author: Anubhab Maity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EcoMambaBlock(nn.Module):
    """
    Lightweight Mamba-inspired block with reduced parameters
    """
    def __init__(self, channels, d_state=16, d_conv=3, expand_factor=2):
        super(LiteMambaBlock, self).__init__()
        hidden_dim = channels * expand_factor
        
        # Projection in with shared normalization
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.proj_in = nn.Conv2d(channels, hidden_dim, kernel_size=1)
        
        # Combined spatial mixing with depth-wise convolution
        self.spatial_mix = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=d_conv, padding=d_conv//2,
            groups=hidden_dim
        )
        
        # Simplified state space parameters with smaller state dimension
        self.ss_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # Single parameter for simplified state evolution
        self.A = nn.Parameter(torch.randn(1, hidden_dim, 1, 1))
        
        # Projection out
        self.proj_out = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        
        # Initialize parameters
        with torch.no_grad():
            self.A.data = -torch.abs(self.A.data)
        
    def forward(self, x):
        residual = x
        
        # Normalization and projection
        x = self.norm(x)
        x = self.proj_in(x)
        
        # Spatial mixing
        x = self.spatial_mix(x)
        x = F.gelu(x)  # GELU is more parameter-efficient than SiLU
        
        # Simplified state space mechanism
        ss_feat = self.ss_proj(x)
        state = ss_feat * torch.sigmoid(self.A)  # Simplified state update
        
        # Projection out
        x = self.proj_out(state)
        
        # Residual connection
        return x + residual


class EfficientAttention(nn.Module):
    """
    Lightweight attention mechanism that combines spatial and channel attention
    """
    def __init__(self, channels, reduction=8):
        super(EfficientAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Use a single shared MLP for channel attention
        self.channel_gate = nn.Sequential(
            nn.Linear(channels, max(8, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, channels // reduction), channels),
            nn.Sigmoid()
        )
        
        # Lightweight spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.channel_gate(y).view(b, c, 1, 1)
        x_channel = x * y
        
        # Spatial attention - use avg and max pooling along channel dimension
        spatial_avg = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_attention = self.spatial_gate(spatial)
        
        return x_channel * spatial_attention



class LiteEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_mamba=False):
        super(LiteEncoderBlock, self).__init__()
        
        # First block using depthwise separable conv
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Add Mamba block conditionally - use lightweight version
        self.mamba = LiteMambaBlock(out_channels) if use_mamba else nn.Identity()
        
        # Efficient attention instead of multiple attention mechanisms
        self.attention = EfficientAttention(out_channels) if out_channels >= 32 else nn.Identity()
        
        # Residual connection for better gradient flow
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # Pooling with strided convolution
        self.pool = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        residual = self.residual_conv(x)
        
        x = self.dw_conv(x)
        x = self.mamba(x)
        x = self.attention(x)
        
        # Add residual connection
        x = x + residual
        
        skip = x  # Save for skip connection
        x = self.pool(x)  # Downsample
        
        return x, skip


class LiteDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_mamba=False):
        super(LiteDecoderBlock, self).__init__()
        
        # Upsampling with transposed convolution
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Lightweight attention gate for skip connections
        self.attention_gate = nn.Sequential(
            nn.Conv2d(in_channels//2 + skip_channels, skip_channels, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel adjustment layer
        self.adjust = nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=1)
        
        # Mamba block for sequence modeling
        self.mamba = LiteMambaBlock(out_channels) if use_mamba else nn.Identity()
        
        # Efficient attention
        self.attention = EfficientAttention(out_channels) if out_channels >= 32 else nn.Identity()
        
        # Residual connection
        self.residual_conv = nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=1)
        
    def forward(self, x, skip):
        # Upsample decoder features
        x = self.upsample(x)
        
        # Handle dimension mismatch
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
            
        # Create attention mask for skip connection
        concat_features = torch.cat([x, skip], dim=1)
        attention_mask = self.attention_gate(concat_features)
        attended_skip = skip * attention_mask
        
        # Save for residual connection
        concat = torch.cat([x, attended_skip], dim=1)
        residual = self.residual_conv(concat)
        
        # Process through decoder blocks
        x = self.adjust(concat)
        x = self.mamba(x)
        x = self.attention(x)
        
        # Add residual
        x = x + residual
        
        return x


class EcoMambaNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_classes=None, base_channels=16, deep_supervision=False):
        super(LightweightMambaUNet, self).__init__()
        
        # Output channels based on segmentation task
        final_out = num_classes if num_classes and num_classes > 1 else out_channels
        
        # Feature channels in each stage - reduced base channels
        c0 = base_channels // 2  # 8
        c1 = base_channels      # 16
        c2 = base_channels * 2  # 32
        c3 = base_channels * 4  # 64
        c4 = base_channels * 8  # 128
        
        # Initial feature extraction - simplified
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=c1),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks - gradually increase Mamba usage for deeper layers
        self.encoder1 = LiteEncoderBlock(c1, c1, use_mamba=False)
        self.encoder2 = LiteEncoderBlock(c1, c2, use_mamba=False)
        self.encoder3 = LiteEncoderBlock(c2, c3, use_mamba=True)
        self.encoder4 = LiteEncoderBlock(c3, c4, use_mamba=True)
        
        # Bottleneck with single Mamba block
        self.bottleneck = LiteMambaBlock(c4)
        
        # Decoder blocks with selective Mamba usage
        self.decoder1 = LiteDecoderBlock(c4, c4, c3, use_mamba=True)
        self.decoder2 = LiteDecoderBlock(c3, c3, c2, use_mamba=False)
        self.decoder3 = LiteDecoderBlock(c2, c2, c1, use_mamba=False)
        
        # Deep supervision - optional and simplified
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.deep_outputs = nn.ModuleList([
                nn.Conv2d(c1, final_out, kernel_size=1),
                nn.Conv2d(c2, final_out, kernel_size=1),
                nn.Conv2d(c3, final_out, kernel_size=1)
            ])
        
        # Final convolution - simplified
        self.final_conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, final_out, kernel_size=1)
        )
        
        # Activation function depends on the task
        self.final_activation = nn.Sigmoid() if final_out == 1 else nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Store the original input size
        input_size = x.size()[2:]
        
        # Initial convolution
        x0 = self.init_conv(x)
        
        # Encoder path
        x1, skip1 = self.encoder1(x0)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)
        
        # Bottleneck
        bottleneck = self.bottleneck(x4)
        
        # Decoder path
        d1 = self.decoder1(bottleneck, skip4)
        d2 = self.decoder2(d1, skip3)
        d3 = self.decoder3(d2, skip2)
        
        # Final convolution
        x = self.final_conv(d3)
        
        # Ensure output is at original input resolution
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        x = self.final_activation(x)
        
        return x
