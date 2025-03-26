import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# -------------------------------
# 1. CNN-ViT for Regression
# -------------------------------
class PatchEmbed3D(nn.Module):
    """
    Embeds 3D patches into a higher-dimensional space.
    For an input of shape [B, in_channels, D, H, W] and patch_size (4,16,16),
    the output will have shape [B, N, embed_dim] with N = (D//4)*(H//16)*(W//16).
    """
    def __init__(self, in_channels=1, patch_size=(4, 16, 16), embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.channel_mixer = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)               # [B, embed_dim, D', H', W']
        x = self.channel_mixer(x)      # [B, embed_dim, D', H', W']
        # Save the grid dimensions for later use in the transformer branch
        self.grid_dims = x.shape[2:5]   # (D', H', W')
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # [B, N, embed_dim]
        return x

# -------------------------------
# 2. Channel Attention (Squeeze-and-Excitation)
# -------------------------------
class ChannelAttention(nn.Module):
    """
    Applies channel-wise attention.
    """
    def __init__(self, embed_dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // reduction, embed_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, N, C] -> [B, C, N] -> [B, C, 1]
        y = self.avg_pool(x.transpose(1, 2))  # [B, C, 1]
        y = y.view(x.size(0), x.size(2))        # [B, C]
        y = self.fc(y).view(x.size(0), 1, x.size(2))  # [B, 1, C]
        return x * y

# -------------------------------
# 3. Transformer Encoder
# -------------------------------
class TransformerEncoder(nn.Module):
    """
    Standard Transformer encoder using PyTorch's nn.TransformerEncoderLayer.
    """
    def __init__(self, embed_dim=256, num_heads=8, depth=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu'
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, N, C] --> Transformer expects [N, B, C]
        x = x.permute(1, 0, 2)  # [N, B, C]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.permute(1, 0, 2)  # [B, N, C]
        return x

# -------------------------------
# 4. Dual-Branch CNN3D (No U-Net Decoder)
# -------------------------------
class DualBranchCNN3D(nn.Module):
    """
    A dual-branch 3D CNN that processes T1CE and T2-FLAIR independently at
    multiple scales, then fuses them for a global representation.
    """
    def __init__(self, out_channels=256, features=[32, 64, 128, 256]):
        super().__init__()
        self.branch_t1 = self._make_branch(1, features)
        self.branch_t2 = self._make_branch(1, features)

        # Fuse final feature maps (concatenate channels, then 1×1×1 conv).
        self.fuse_conv = nn.Conv3d(features[-1]*2, features[-1]*2, kernel_size=1)

        self.post_fusion_block = nn.Sequential(
            nn.Conv3d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(features[-1]*2),
            nn.ReLU(inplace=True)
        )

        # Final 1×1 conv to get the desired out_channels
        self.final_conv = nn.Conv3d(features[-1]*2, out_channels, kernel_size=1)

        # Global adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.initialize_weights()

    def _make_branch(self, in_channels, features):
        """
        Builds multiple 'down' blocks:
          [Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool]
        for each specified feature size in `features`.
        """
        layers = []
        for f in features:
            block = nn.Sequential(
                nn.Conv3d(in_channels, f, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(f),
                nn.ReLU(inplace=True),
                nn.Conv3d(f, f, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(f),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2)
            )
            layers.append(block)
            in_channels = f
        return nn.ModuleList(layers)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 2, D, H, W]
               - channel 0: T1CE
               - channel 1: T2-FLAIR
        Returns:
            features: [B, out_channels]
        """
        x_t1 = x[:, 0:1, :, :, :]
        x_t2 = x[:, 1:2, :, :, :]

        for block in self.branch_t1:
            x_t1 = block(x_t1)
        for block in self.branch_t2:
            x_t2 = block(x_t2)

        # Fuse final feature maps
        x_fused = torch.cat([x_t1, x_t2], dim=1)
        x_fused = self.fuse_conv(x_fused)
        x_fused = self.post_fusion_block(x_fused)

        x_out = self.final_conv(x_fused)
        features = self.adaptive_pool(x_out).view(x_out.size(0), -1)
        return features

# -------------------------------
# 5. Vision Transformer for 3D Data (1 Channel)
# -------------------------------
class VisionTransformer3D_1Channel(nn.Module):
    """
    Vision Transformer adapted for a single 3D imaging channel.
    For an input of shape [B, 1, D, H, W] and patch_size (4,16,16),
    the number of patches is computed as:
        num_patches = (D//4) * (H//16) * (W//16)
    This module outputs a feature vector of size `feature_dim` for fusion in the regression model.
    """
    def __init__(self, in_channels=1, patch_size=(4, 16, 16), embed_dim=512,
                 num_heads=12, depth=12, feature_dim=512, dropout=0.3,
                 default_input_shape=(100, 256, 256)):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, patch_size, embed_dim)
        # Default grid size based on assumed default input shape
        D, H, W = default_input_shape
        self.default_grid_size = (D // patch_size[0], H // patch_size[1], W // patch_size[2])
        num_patches = self.default_grid_size[0] * self.default_grid_size[1] * self.default_grid_size[2]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=4.0,
            dropout=dropout
        )
        self.channel_attention = ChannelAttention(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # Here, the classifier head is renamed to output a feature vector of dimension `feature_dim`
        self.classifier = nn.Linear(embed_dim, feature_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 1, D, H, W]
        Returns:
            out: [B, feature_dim]
        """
        B = x.size(0)
        # Obtain patch embeddings
        x_patch = self.patch_embed(x)  # [B, N, embed_dim]
        # Retrieve grid dimensions from the patch embedding layer
        D_new, H_new, W_new = self.patch_embed.grid_dims

        # Rearrange the patch tokens back into grid form (for positional interpolation)
        x_seq = x_patch.view(B, D_new, H_new, W_new, -1)
        x_seq = rearrange(x_seq, 'b d h w c -> b (d h w) c')

        # Interpolate the positional embedding if grid size differs from default
        if (D_new, H_new, W_new) != self.default_grid_size:
            # Separate class token and patch tokens
            cls_token = self.pos_embed[:, :1, :]
            pos_tokens = self.pos_embed[:, 1:, :]  # [1, old_num_patches, embed_dim]
            # Reshape patch tokens into grid of shape [1, D_old, H_old, W_old, embed_dim]
            D_old, H_old, W_old = self.default_grid_size
            pos_tokens = pos_tokens.view(1, D_old, H_old, W_old, -1)
            # Permute to [1, embed_dim, D_old, H_old, W_old] for interpolation
            pos_tokens = pos_tokens.permute(0, 4, 1, 2, 3)
            # Interpolate to new grid size (trilinear interpolation)
            pos_tokens = F.interpolate(pos_tokens, size=(D_new, H_new, W_new), mode='trilinear', align_corners=False)
            # Permute back and flatten to [1, new_num_patches, embed_dim]
            pos_tokens = pos_tokens.permute(0, 2, 3, 4, 1).reshape(1, D_new * H_new * W_new, -1)
            pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
        else:
            pos_embed = self.pos_embed

        # Prepare class token and concatenate with patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x_final = torch.cat((cls_tokens, x_seq), dim=1)  # [B, N+1, embed_dim]
        x_final = x_final + pos_embed
        x_final = self.dropout(x_final)
        x_final = self.transformer(x_final)
        cls_out = x_final[:, 0, :]  # [B, embed_dim]
        # Obtain feature vector
        out = self.classifier(cls_out)
        return out

# -------------------------------
# 6. Clinical Data Processor (MLP)
# -------------------------------
class ClinicalDataProcessor(nn.Module):
    """
    Processes clinical data via a simple MLP.
    """
    def __init__(self, input_dim=15, embed_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------------
# 7. Main Model: CNNViT for Regression
# -------------------------------
class CNNViT(nn.Module):
    """
    Combines:
      - A dual-branch 3D CNN for MR T1CE and T2-FLAIR (channels 0 and 1),
      - A Vision Transformer for a third imaging channel (e.g. another modality) that outputs a feature vector,
      - And an MLP for clinical data.
    The features from all branches are fused to produce a final regression output.
    """
    def __init__(self, output_dim=1, clinical_dim=15):
        super().__init__()
        # Dual-branch 3D CNN for T1CE and T2-FLAIR
        self.dual_cnn = DualBranchCNN3D(out_channels=512, features=[64, 128, 256, 512])

        # Vision Transformer for the third channel (e.g. a different modality)
        self.vision_transformer = VisionTransformer3D_1Channel(
            in_channels=1,
            patch_size=(4, 16, 16),
            embed_dim=512,
            num_heads=16,
            depth=12,
            feature_dim=512,
            dropout=0.3,
            default_input_shape=(100, 256, 256)
        )

        # MLP for clinical data
        self.clinical_processor = ClinicalDataProcessor(
            input_dim=clinical_dim,
            embed_dim=512
        )

        # Normalization layers for stability
        self.cnn_norm = nn.LayerNorm(512)
        self.transformer_norm = nn.LayerNorm(512)
        self.clinical_norm = nn.LayerNorm(512)

        # Final regressor: combines features from CNN, Transformer, and clinical data
        self.regressor = nn.Sequential(
            nn.Linear(512 + 512 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, output_dim)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, clinical_data):
        """
        Args:
            x: [B, 3, D, H, W]
               - channels [0,1] = T1CE, T2-FLAIR (for dual_cnn)
               - channel [2]   = third modality (for vision_transformer)
            clinical_data: [B, clinical_dim]
        Returns:
            out: [B, output_dim] (continuous regression prediction)
        """
        # Extract subsets for each module
        x_cnn = x[:, :2, :, :, :]          # T1CE, T2-FLAIR
        x_transformer = x[:, 2:3, :, :, :]   # third modality

        # CNN branch
        cnn_features = self.dual_cnn(x_cnn)
        cnn_features = self.cnn_norm(cnn_features)

        # Vision Transformer branch
        transformer_features = self.vision_transformer(x_transformer)
        transformer_features = self.transformer_norm(transformer_features)

        # Clinical branch
        clinical_features = self.clinical_processor(clinical_data)
        clinical_features = self.clinical_norm(clinical_features)

        # Fuse all modality features
        combined_features = torch.cat(
            [cnn_features, transformer_features, clinical_features], dim=1
        )
        out = self.regressor(combined_features)
        return out

# -------------------------------
# 8. Factory Function for CNNViT Regression Model
# -------------------------------
def cnn_vit_model(output_dim=1, clinical_dim=15):
    """
    Returns an instance of the CNNViT model for regression.
    """
    return CNNViT(output_dim=output_dim, clinical_dim=clinical_dim)


# -----------------------------------------------
# 9. Channel-Separated CNN Model for regression
# -----------------------------------------------

class ChannelSeparatedCNN(nn.Module):
    """
    CNN model with separate Conv3d layers for each channel, which first extracts features for each channel and then combines them for further processing.
    This version uses adaptive pooling to handle varying input sizes.
    """
    def __init__(self, in_channels=3, clinical_dim=15, target_output_size=(6, 16, 16)):
        """
        Args:
            in_channels (int): Number of channels in the image input.
            clinical_dim (int): Dimension of the clinical data.
            target_output_size (tuple): Target output size (D, H, W) for adaptive pooling after convolutions.
        """
        super(ChannelSeparatedCNN, self).__init__()
        # Create separate Conv3d layers for each channel
        self.channel_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Dropout3d(0.2)
            ) for _ in range(in_channels)
        ])
        # After concatenation, total channels = 32 * in_channels
        self.conv1 = nn.Conv3d(32 * in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout3d(0.3)
        # Adaptive pooling for fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool3d(target_output_size)
        self.flattened_size = 128 * target_output_size[0] * target_output_size[1] * target_output_size[2]
        # Process clinical data
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512)
        )
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size + 512, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout_fc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x, clinical_data):
        batch_size = x.size(0)
        channel_features = []
        for i, conv in enumerate(self.channel_convs):
            channel = x[:, i:i+1, :, :, :]
            feature = conv(channel)
            channel_features.append(feature)
        x = torch.cat(channel_features, dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        x = self.adaptive_pool(x)
        x = x.view(batch_size, -1)
        clinical_features = self.clinical_processor(clinical_data)
        combined = torch.cat([x, clinical_features], dim=1)
        x = F.relu(self.bn_fc1(self.fc1(combined)))
        x = self.dropout_fc1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def channel_separated_cnn_model(in_channels=3, clinical_dim=21, target_output_size=(6, 16, 16)):
    """
    Factory function to create a ChannelSeparatedCNN model.

    Args:
        in_channels (int): Number of channels in the image input.
        clinical_dim (int): Dimension of the clinical data.
        target_output_size (tuple): Target output size (D, H, W) after adaptive pooling.
    """
    model = ChannelSeparatedCNN(in_channels=in_channels, clinical_dim=clinical_dim,
                                 target_output_size=target_output_size)
    model.initialize_weights()
    return model


# -----------------------------------------------
# 10. Depthwise Separable 3D CNN Model for Regression
# -----------------------------------------------
class Shared3DCNN(nn.Module):
    """
    CNN block for processing a single channel.
    This is used for the first two channels (shared weights),
    but with fewer channels than the original.
    """
    def __init__(self):
        super(Shared3DCNN, self).__init__()
        # Reduced output channels to lower memory usage
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm3d(64)

        self.pool  = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout3d(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        return x

class Separate3DCNN(nn.Module):
    """
    Separate CNN block for processing the third channel.
    Reduced in the same way as Shared3DCNN.
    """
    def __init__(self):
        super(Separate3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm3d(64)

        self.pool  = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout3d(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        return x

class DepthwiseSeparable3DCNN(nn.Module):
    """
    3D CNN that processes the first two channels with a shared CNN and the third with a separate CNN.
    Clinical data is combined after the image processing part.
    Expected input: (B, 3, D, H, W)
    """
    def __init__(self, clinical_dim=15, input_shape=(100, 256, 256)):
        super(DepthwiseSeparable3DCNN, self).__init__()
        # Shared CNN for channel 0 and 1
        self.shared_cnn = Shared3DCNN()
        # Separate CNN for channel 2
        self.separate_cnn = Separate3DCNN()

        # Dummy to calculate the shape of the CNN output
        dummy_input = torch.zeros(1, 1, *input_shape)  # one channel

        # Output from the shared CNN (used for two channels)
        dummy_out_shared = self.shared_cnn(dummy_input)
        shared_out_size = dummy_out_shared.view(1, -1).size(1)

        # Output from the separate CNN (third channel)
        dummy_out_separate = self.separate_cnn(dummy_input)
        separate_out_size = dummy_out_separate.view(1, -1).size(1)

        # Total output size from all image channels
        self.cnn_out_size = shared_out_size * 2 + separate_out_size

        # Clinical processor (reduced size)
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256)
        )

        # Fully connected layers (reduced size)
        self.fc1 = nn.Linear(self.cnn_out_size + 256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, clinical_data):
        # x is expected to have shape (B, 3, D, H, W)
        B = x.size(0)

        # Channels 0 and 1 through the shared CNN
        x0 = self.shared_cnn(x[:, 0:1, ...])
        x1 = self.shared_cnn(x[:, 1:2, ...])

        # Channel 2 through the separate CNN
        x2 = self.separate_cnn(x[:, 2:3, ...])

        # Combine outputs along the channel axis
        x = torch.cat([x0, x1, x2], dim=1)
        x = x.view(B, -1)

        # Clinical data
        clinical_features = self.clinical_processor(clinical_data)

        # Combine image and clinical features
        combined = torch.cat([x, clinical_features], dim=1)

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(combined)))
        x = self.dropout_fc1(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)

        x = self.fc3(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def depthwise_separable_3dcnn_model(clinical_dim=15, input_shape=(100, 256, 256)):
    model = DepthwiseSeparable3DCNN(clinical_dim=clinical_dim, input_shape=input_shape)
    model.initialize_weights()
    return model


# -----------------------------------------------
# 11. 3D ResNet for Single-Channel Input
class BasicBlock3D(nn.Module):
    """A simplified 3D ResNet Basic Block with two 3x3x3 convolutions."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        """
        Args:
            in_planes  (int): number of input channels
            planes     (int): number of output channels
            stride     (int): stride of the first convolution
            downsample (nn.Module or None): downsampling layer to match dims
        """
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(planes)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If there is a downsample layer, apply it to the identity
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


################################################################################
# 2) A small 3D ResNet for single-channel input
################################################################################
class SmallResNet3D(nn.Module):
    """
    Minimal 3D ResNet-like network that takes a single 3D channel and produces a feature map.
    We'll keep the channel sizes small (16 -> 32 -> 64) to limit memory usage.
    """
    def __init__(self):
        super(SmallResNet3D, self).__init__()

        # Initial convolution and batch norm
        self.in_planes = 16
        self.conv1 = nn.Conv3d(1, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(self.in_planes)
        self.relu  = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # ResNet layers
        self.layer1 = self._make_layer(16, 1, stride=1)  # 1 block
        self.pool2  = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer2 = self._make_layer(32, 1, stride=1)  # 1 block
        self.pool3  = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer3 = self._make_layer(64, 1, stride=1)  # 1 block
        self.pool4  = nn.MaxPool3d(kernel_size=2, stride=2)

        # Optional dropout
        self.dropout = nn.Dropout3d(0.3)

    def _make_layer(self, planes, num_blocks, stride=1):
        """
        Create a layer that may have multiple BasicBlock3Ds, but here we'll keep it at 1 for simplicity.
        """
        downsample = None
        if stride != 1 or self.in_planes != planes:
            # Downsample if shape mismatch
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

        layers = []
        layers.append(BasicBlock3D(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        # If you wanted more blocks, you would loop for the rest. We keep it at 1 here.
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        # layer1
        x = self.layer1(x)
        x = self.pool2(x)
        x = self.dropout(x)

        # layer2
        x = self.layer2(x)
        x = self.pool3(x)
        x = self.dropout(x)

        # layer3
        x = self.layer3(x)
        x = self.pool4(x)
        x = self.dropout(x)

        return x


################################################################################
# 3) Shared and separate "ResNets" for your specialized multi-channel input
################################################################################
class SharedResNet3D(nn.Module):
    """
    This will be used for channels 0 and 1, sharing the same weights.
    """
    def __init__(self):
        super(SharedResNet3D, self).__init__()
        self.model = SmallResNet3D()

    def forward(self, x):
        return self.model(x)

class SeparateResNet3D(nn.Module):
    """
    This will be used for the third channel (channel 2).
    """
    def __init__(self):
        super(SeparateResNet3D, self).__init__()
        self.model = SmallResNet3D()

    def forward(self, x):
        return self.model(x)


################################################################################
# 4) Top-level model that merges the three channels + clinical data
################################################################################
class DepthwiseSeparable3DResNet(nn.Module):
    """
    Similar to your DepthwiseSeparable3DCNN, but using a ResNet-like backbone.
    Channels 0 and 1 -> shared ResNet
    Channel 2 -> separate ResNet
    """
    def __init__(self, clinical_dim=15, input_shape=(100, 256, 256)):
        super(DepthwiseSeparable3DResNet, self).__init__()

        # Instantiate the shared and separate nets
        self.shared_net = SharedResNet3D()
        self.separate_net = SeparateResNet3D()

        # Test on dummy input to figure out output feature sizes
        dummy_input = torch.zeros(1, 1, *input_shape)  # single channel

        # Check feature size of shared net (for channel 0 or 1)
        with torch.no_grad():
            shared_out = self.shared_net(dummy_input)
        shared_out_size = shared_out.view(1, -1).size(1)

        # Check feature size of separate net (for channel 2)
        with torch.no_grad():
            separate_out = self.separate_net(dummy_input)
        separate_out_size = separate_out.view(1, -1).size(1)

        # Shared net is used for two channels => multiply by 2
        self.cnn_out_size = 2 * shared_out_size + separate_out_size

        # Clinical processor
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256)
        )

        # Combine them in fully connected layers
        self.fc1 = nn.Linear(self.cnn_out_size + 256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 1)

        # Initialize
        self.initialize_weights()

    def forward(self, x, clinical_data):
        """
        x shape: (B, 3, D, H, W)
        clinical_data shape: (B, clinical_dim)
        """
        B = x.size(0)

        # Channels 0 and 1 through the shared ResNet
        x0 = self.shared_net(x[:, 0:1, ...])
        x1 = self.shared_net(x[:, 1:2, ...])

        # Channel 2 through the separate ResNet
        x2 = self.separate_net(x[:, 2:3, ...])

        # Concatenate along channel dim
        feat = torch.cat([x0, x1, x2], dim=1)
        feat = feat.view(B, -1)

        # Process clinical data
        clinical_features = self.clinical_processor(clinical_data)

        # Combine
        combined = torch.cat([feat, clinical_features], dim=1)

        # FC layers
        out = F.relu(self.bn_fc1(self.fc1(combined)))
        out = self.dropout_fc1(out)

        out = F.relu(self.bn_fc2(self.fc2(out)))
        out = self.dropout_fc2(out)

        out = self.fc3(out)
        return out

    def initialize_weights(self):
        # You can also initialize the ResNet submodules if desired
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



def depthwise_separable_3dresnet_model(clinical_dim=15, input_shape=(100, 256, 256)):
    model = DepthwiseSeparable3DResNet(clinical_dim=clinical_dim,
                                       input_shape=input_shape)
    return model

class SimpleCNN(nn.Module):
    """
    A refined CNN model for regression that handles various input shapes.
    Example input shapes: (3, 50, 128, 128) or (3, 100, 256, 256).
    """
    def __init__(self, input_shape, clinical_dim=15):
        """
        Args:
            input_shape (tuple): The shape of the input (channels, depth, height, width).
            clinical_dim (int): The dimension of the clinical data.
        """
        super(SimpleCNN, self).__init__()

        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv3d(input_shape[0], 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(512)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout3d(0.3)

        # Compute the flattened size by performing a dummy forward-pass
        self.flattened_size = self._get_conv_output(input_shape)

        # Processor for clinical data
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size + 512, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 1)

    def _get_conv_output(self, shape):
        """
        Performs a dummy forward-pass to compute the number of features after the convolutional layers.
        """
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = self.dropout_conv(x)

            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = self.dropout_conv(x)

            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = self.dropout_conv(x)

            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)
            x = self.dropout_conv(x)

            n_size = x.view(1, -1).size(1)
        return n_size

    def forward(self, x, clinical_data):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = x.view(x.size(0), -1)  # Flatten

        # Process clinical data
        clinical_features = self.clinical_processor(clinical_data)

        # Combine image and clinical features
        combined = torch.cat([x, clinical_features], dim=1)

        # Fully connected layers with ReLU, BatchNorm, and Dropout
        x = F.relu(self.bn_fc1(self.fc1(combined)))
        x = self.dropout_fc1(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)

        x = self.fc3(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def simple_cnn_model(input_shape=(3, 50, 128, 128), clinical_dim=21):
    """
    Factory function to create a SimpleCNN instance.

    Args:
        input_shape (tuple): For example, (3, 50, 128, 128) or (3, 100, 256, 256).
        clinical_dim (int): The dimension of the clinical data.
    """
    model = SimpleCNN(input_shape=input_shape, clinical_dim=clinical_dim)
    model.initialize_weights()
    return model