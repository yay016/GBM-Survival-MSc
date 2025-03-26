import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DepthwiseSeparableConv3D(nn.Module):
    """
    Implements a Depthwise Separable Convolution for 3D data.
    First, a depthwise convolution is applied (one filter per input channel),
    followed by a pointwise convolution (1x1x1) to mix the channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the kernel.
        stride (int or tuple): Stride.
        padding (int or tuple): Padding.
        bias (bool): Whether to include a bias term.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv3D, self).__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # One filter per channel
            bias=bias
        )
        self.pointwise = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseCNN3D(nn.Module):
    """
    A simple 3D CNN model with depthwise separable convolutions for classification,
    which also includes an MLP for processing clinical data.

    Architecture:
      - Three blocks with Depthwise Separable Convolution, BatchNorm, ReLU, and MaxPool.
      - Global average pooling to reduce spatial dimensions.
      - An MLP for clinical data (input: clinical_dim) mapped to a 128-dimensional space.
      - Combination of image and clinical features before a fully connected classification layer.

    Args:
        in_channels (int): Number of input channels for image data (e.g., 3).
        num_classes (int): Number of output classes.
        clinical_dim (int): Dimension of the clinical data.
    """
    def __init__(self, in_channels=3, num_classes=1, clinical_dim=16):
        super(DepthwiseCNN3D, self).__init__()
        # Image branch
        self.layer1 = nn.Sequential(
            DepthwiseSeparableConv3D(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)  # Reduces spatial resolution
        )
        self.layer2 = nn.Sequential(
            DepthwiseSeparableConv3D(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            DepthwiseSeparableConv3D(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        # Global average pooling to compress to one value per channel
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Clinical MLP: Maps clinical data (clinical_dim) to a 128-dimensional space.
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        # Classifier: Combines image and clinical features (128 + 128 = 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, clinical_data):
        """
        Args:
            x (torch.Tensor): Image input of shape [B, in_channels, D, H, W].
            clinical_data (torch.Tensor): Clinical data of shape [B, clinical_dim].
        Returns:
            torch.Tensor: Classification output of shape [B, num_classes].
        """
        # Image branch
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)   # [B, 128, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]

        # Clinical branch
        clinical_features = self.clinical_processor(clinical_data)  # [B, 128]

        # Combine image and clinical features
        combined_features = torch.cat([x, clinical_features], dim=1)  # [B, 256]
        out = self.classifier(combined_features)  # [B, num_classes]
        return out


def depthwise_cnn_model(in_channels=3, num_classes=1, clinical_dim=16):
    """
    Factory function to create a DepthwiseCNN3D model with clinical data.

    Args:
        in_channels (int): Number of input channels for image data.
        num_classes (int): Number of output classes.
        clinical_dim (int): Dimension of the clinical data.

    Returns:
        An instance of DepthwiseCNN3D.
    """
    return DepthwiseCNN3D(in_channels=in_channels, num_classes=num_classes, clinical_dim=clinical_dim)


class SimpleCNN(nn.Module):
    """
    A simple 3D CNN model for classification that processes 3D MR data and combines
    it with clinical data using a simple MLP. This model uses standard 3D convolutions,
    Batch Normalization, ReLU activations, and MaxPooling to extract image features.

    Architecture:
      - Three convolutional blocks with Conv3d, BatchNorm3d, ReLU, and MaxPool3d.
      - Global average pooling to reduce the spatial dimensions.
      - An MLP for clinical data (input: clinical_dim) mapped to a 128-dimensional space.
      - Combination of image and clinical features before a final fully connected layer.

    Args:
        in_channels (int): Number of input channels for image data (e.g., 2 for T1 and T2).
        num_classes (int): Number of output classes.
        clinical_dim (int): Dimension of the clinical data.
    """
    def __init__(self, in_channels=2, num_classes=1, clinical_dim=16):
        super(SimpleCNN, self).__init__()
        # Image branch using standard Conv3d layers
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        # Global average pooling to get one value per channel
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Clinical branch: maps clinical data to a 128-dimensional feature space
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        # Classifier: combines image and clinical features (128 + 128 = 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, clinical_data):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): 3D image input of shape [B, in_channels, D, H, W].
            clinical_data (torch.Tensor): Clinical data of shape [B, clinical_dim].

        Returns:
            torch.Tensor: Output predictions of shape [B, num_classes].
        """
        # Process image data through convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)  # [B, 128, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]

        # Process clinical data
        clinical_features = self.clinical_processor(clinical_data)  # [B, 128]

        # Concatenate features and classify
        combined_features = torch.cat([x, clinical_features], dim=1)  # [B, 256]
        out = self.classifier(combined_features)
        return out

def simple_cnn_model(in_channels=3, num_classes=1, clinical_dim=16):
    """
    Factory function to create a SimpleCNN model.

    Args:
        in_channels (int): Number of image channels (e.g., 2 for T1 and T2).
        num_classes (int): Number of output classes.
        clinical_dim (int): Dimension of the clinical data.

    Returns:
        SimpleCNN: An instance of the SimpleCNN model.
    """
    return SimpleCNN(in_channels=in_channels, num_classes=num_classes, clinical_dim=clinical_dim)

class SingleChannelCNN(nn.Module):
    """
    CNN branch for a single image channel.

    Input: 3D volume with shape [B, 1, 50, 128, 128]
    Output: A 128-dimensional feature vector for the channel.
    """
    def __init__(self, base_filters=32, out_dim=128):
        super(SingleChannelCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)  # Reduces dimensions
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        # After 3 pooling layers:
        # D: 50 -> 25 -> 12 -> 6, H/W: 128 -> 64 -> 32 -> 16.
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(base_filters * 4, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)  # [B, base_filters*4, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, base_filters*4]
        x = self.fc(x)             # [B, out_dim]
        return x


class ClinicalMLP(nn.Module):
    """
    A multilayer perceptron (MLP) for processing clinical data.

    Input: Clinical data with shape [B, clinical_dim]
    Output: A 128-dimensional feature vector.
    """
    def __init__(self, clinical_dim, hidden_dim=128):
        super(ClinicalMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)


class ChannelwiseCNNModel(nn.Module):
    """
    Channelwise model that creates a separate CNN branch for each image channel
    (3D volumes with dimensions 50×128×128) along with an MLP for clinical data.

    Args:
        num_image_channels (int): Number of image channels.
        clinical_dim (int): Dimension of the clinical data.
        num_classes (int): Number of output classes.
    """
    def __init__(self, num_image_channels=3, clinical_dim=16, num_classes=1):
        super(ChannelwiseCNNModel, self).__init__()
        self.num_image_channels = num_image_channels

        # Create a separate CNN branch for each image channel
        self.branches = nn.ModuleList([
            SingleChannelCNN(base_filters=32, out_dim=128)
            for _ in range(num_image_channels)
        ])

        # MLP for clinical data
        self.clinical_mlp = ClinicalMLP(clinical_dim=clinical_dim, hidden_dim=128)

        # Combined features: from image branch (num_image_channels * 128) + clinical (128)
        total_features = num_image_channels * 128 + 128
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, clinical_data):
        """
        Args:
            x (torch.Tensor): Image input with shape [B, num_image_channels, 50, 128, 128].
            clinical_data (torch.Tensor): Clinical data with shape [B, clinical_dim].
        Returns:
            torch.Tensor: Classification output with shape [B, num_classes].
        """
        branch_features = []
        for i in range(self.num_image_channels):
            # Extract one channel: [B, 1, 50, 128, 128]
            xi = x[:, i:i+1, :, :, :]
            feat = self.branches[i](xi)  # [B, 128]
            branch_features.append(feat)

        # Combine features from all image channels
        image_features = torch.cat(branch_features, dim=1)  # [B, num_image_channels * 128]

        # Process clinical data with the MLP
        clinical_features = self.clinical_mlp(clinical_data)  # [B, 128]

        # Combine image and clinical features
        combined_features = torch.cat([image_features, clinical_features], dim=1)  # [B, total_features]
        out = self.classifier(combined_features)
        return out


def channelwise_cnn_model(num_image_channels=3, clinical_dim=16, num_classes=1):
    """
    Factory function to instantiate a ChannelwiseCNNModel.

    Args:
        num_image_channels (int): Number of image channels.
        clinical_dim (int): Dimension of the clinical data.
        num_classes (int): Number of output classes.
    Returns:
        An instance of ChannelwiseCNNModel.
    """
    return ChannelwiseCNNModel(num_image_channels=num_image_channels,
                               clinical_dim=clinical_dim,
                               num_classes=num_classes)
    
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



def depthwise_separable_3dresnet_model(clinical_dim=15, input_shape=(50, 128, 128), num_classes=1):
    model = DepthwiseSeparable3DResNet(clinical_dim=clinical_dim,
                                       input_shape=input_shape)
    return model
    
    
# -------------------------------
# 1. 3D Patch Embedding
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
    """
    def __init__(self, in_channels=1, patch_size=(4, 16, 16), embed_dim=512,
                 num_heads=12, depth=12, num_classes=512, dropout=0.3,
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
        self.classifier = nn.Linear(embed_dim, num_classes)

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
            out: [B, num_classes]
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
# 7. Main Model: CNNViT
# -------------------------------
class CNNViT(nn.Module):
    """
    Combines:
      - A dual-branch 3D CNN for MR T1CE and T2-FLAIR (channels 0 and 1),
      - A Vision Transformer for a third imaging channel (e.g. another modality),
      - And an MLP for clinical data.
    The features from all branches are fused for final classification.
    """
    def __init__(self, num_classes=1, clinical_dim=15):
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
            num_classes=512,
            dropout=0.3,
            default_input_shape=(50, 128, 128)
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

        # Final classifier: combines features from CNN, Transformer, and clinical data
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.classifier.modules():
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
            out: [B, num_classes]
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
        out = self.classifier(combined_features)
        return out

# -------------------------------
# 8. Factory Function for CNNViT
# -------------------------------
def cnn_vit(num_classes=1, clinical_dim=15):
    """
    Returns an instance of the CNNViT model.
    """
    return CNNViT(num_classes=num_classes, clinical_dim=clinical_dim)