import numpy as np
import torch
from torch import nn
import torchvision
from .utils import LinearReduction, FixedReduction
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def RandomForest():
    classifier = Pipeline([
        ("normalise",StandardScaler(),),
        ("randomforest",RandomForestClassifier(n_estimators=500, min_samples_leaf=10))
    ])
    return classifier

def SVM():
    classifier = Pipeline([
        ("normalise", StandardScaler(),),
        ("kernel_map", Nystroem(kernel='rbf', gamma=1e-4, n_components=500),),
        ("svm", LinearSVC(C=1.0, )),
    ])
    return classifier

class MLP(nn.Module):
    def __init__(self, input_dim, reduce_dim, n_classes, reduce_method, dropout_p=0.5):
        super().__init__()

        # reduction
        # input processing and dimensionality reduction
        if reduce_method == 'fixed':
            self.input_processing = FixedReduction(input_dim)
        elif reduce_method == 'linear':
            self.input_processing = LinearReduction(input_dim, reduce_dim)

        # Normalisation Layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

        # Fc Layers
        self.fc1 = nn.Linear(reduce_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_classes)

        # Additional kit
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)

        self.classifier = nn.Sequential(
            self.fc1,
            self.activation,
            self.bn1,
            self.fc2,
            self.activation,
            self.bn2,
            self.fc3,
        )

    def forward(self, x):
        inputs = self.input_processing(x)
        logits = self.classifier(inputs.flatten(1))
        return logits

class patch3_cnn(nn.Module):
    def __init__(self, input_dim, reduce_dim, n_classes, reduce_method, dropout_p=0.5):
        super().__init__()

        # reduction
        # input processing and dimensionality reduction
        if reduce_method == 'fixed':
            self.input_processing = FixedReduction(input_dim)
        elif reduce_method == 'linear':
            self.input_processing = LinearReduction(input_dim, reduce_dim)

        # Convolution layers
        self.conv1 = nn.Conv2d(reduce_dim, 512, 3, stride=1, padding=0, padding_mode='reflect')
        self.conv2 = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(512, 512, 1, stride=1, padding=0)

        # Normalisation Layers
        self.input_norm = nn.BatchNorm2d(input_dim)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)

        # Fc Layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_classes)

        # Additional kit
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)

        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.activation,
            self.bn1,
            self.conv2,
            self.activation,
            self.bn2,
            self.conv3,
            self.activation,
            self.bn3,
        )

        self.classifier = nn.Sequential(
            self.fc1,
            self.activation,
            self.bn4,
            self.dropout,
            self.fc2,
            self.activation,
            self.bn5,
            self.fc3
        )

    def forward(self, x):
        inputs = self.input_processing(x)
        features = self.feature_extractor(inputs)
        logits = self.classifier(features.flatten(1))
        return logits

class patch25_cnn(nn.Module):
    def __init__(self, input_dim, reduce_dim, n_classes, reduce_method, dropout_p=0.5):
        super().__init__()

        # input processing and dimensionality reduction
        if reduce_method == 'fixed':
            self.input_processing = FixedReduction(input_dim)
        elif reduce_method == 'linear':
            self.input_processing = LinearReduction(input_dim, reduce_dim)

        # Convolution layers
        self.conv1 = nn.Conv2d(reduce_dim, 32, 3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1, padding_mode='reflect')

        # Normalisation Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)

        # Fc Layers
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_classes)

        # Additional kit
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)

        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.activation,
            self.pool,
            self.bn1,
            self.conv2,
            self.activation,
            self.bn2,
            self.conv3,
            self.activation,
            self.pool,
            self.bn3,
        )

        self.classifier = nn.Sequential(
            self.fc1,
            self.activation,
            self.bn4,
            self.dropout,
            self.fc2,
            self.activation,
            self.bn5,
            self.fc3
        )

    def forward(self, x):
        inputs = self.input_processing(x)
        features = self.feature_extractor(inputs)
        logits = self.classifier(features.flatten(1))
        return logits

class patch101_cnn(nn.Module):
    def __init__(self, input_dim, reduce_dim, n_classes, reduce_method, dropout_p=0.5):
        super().__init__()

        # input processing and dimensionality reduction
        if reduce_method == 'fixed':
            self.input_processing = FixedReduction(input_dim)
        elif reduce_method == 'linear':
            self.input_processing = LinearReduction(input_dim, reduce_dim)

        # Convolution layers
        self.conv1 = nn.Conv2d(reduce_dim, 32, 5, stride=2, padding=0, padding_mode='reflect')
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1, padding_mode='reflect')

        # Normalisation Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)

        # Fc Layers
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_classes)

        # Additional kit
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)

        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.activation,
            self.pool,
            self.bn1,
            self.conv2,
            self.activation,
            self.pool,
            self.bn2,
            self.conv3,
            self.activation,
            self.pool,
            self.bn3,
        )

        self.classifier = nn.Sequential(
            self.fc1,
            self.activation,
            self.bn4,
            self.dropout,
            self.fc2,
            self.activation,
            self.bn5,
            self.fc3
        )

    def forward(self, x):
        inputs = self.input_processing(x)
        features = self.feature_extractor(inputs)
        logits = self.classifier(features.flatten(1))
        return logits

class patch_multiscale(nn.Module):
    def __init__(self, input_dim, reduce_dim, n_classes, reduce_method, dropout_p=0.5):
        super().__init__()

        self.patch_3px = patch3_cnn(input_dim, reduce_dim, n_classes, reduce_method, dropout_p)
        self.patch_3px.classifier = self.patch_3px.classifier[:4]

        self.patch_101px = patch101_cnn(input_dim, reduce_dim, n_classes, reduce_method, dropout_p)
        self.patch_101px.classifier = self.patch_101px.classifier[:4]

        # Normalisation Layers
        self.bn5 = nn.BatchNorm1d(256)

        # Fc Layers
        self.fc2 = nn.Linear(256 * 2, 256)
        self.fc3 = nn.Linear(256, n_classes)

        # Additional kit
        self.activation = nn.GELU()

        self.feature_fusion_classifier = nn.Sequential(
            self.fc2,
            self.activation,
            self.bn5,
            self.fc3
        )

    def forward(self, x):
        patch_3px_features = self.patch_3px(x[:,:,49:-49,49:-49])
        patch_101px_features = self.patch_101px(x)
        logits = self.feature_fusion_classifier(torch.cat([patch_3px_features, patch_101px_features],dim=1))
        return logits

# based on the implementation in https://github.com/milesial/Pytorch-UNet
class UNet(nn.Module):
    def __init__(self, input_dim, reduce_dim, n_classes, reduce_method, bilinear=False): # todo mention pytorch unet
        super(UNet, self).__init__()

        # input processing and dimensionality reduction
        if reduce_method == 'fixed':
            self.input_processing = FixedReduction(input_dim)
        elif reduce_method == 'linear':
            self.input_processing = LinearReduction(input_dim, reduce_dim)

        self.n_channels = reduce_dim
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (self.DoubleConv(reduce_dim, 64))
        self.down1 = (self.Down(64, 128))
        self.down2 = (self.Down(128, 256))
        self.down3 = (self.Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (self.Down(512, 1024 // factor))
        self.up1 = (self.Up(1024, 512 // factor, bilinear))
        self.up2 = (self.Up(512, 256 // factor, bilinear))
        self.up3 = (self.Up(256, 128 // factor, bilinear))
        self.up4 = (self.Up(128, 64, bilinear))
        self.outc = (self.OutConv(64, n_classes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        inputs = self.input_processing(x)

        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    class DoubleConv(nn.Module):
        """(convolution => [BN] => ReLU) * 2"""

        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)

    class OutConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    class Up(nn.Module):
        """Upscaling then double conv"""

        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()

            # if bilinear, use the normal convolutions to reduce the number of channels
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = UNet.DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = UNet.DoubleConv(in_channels, out_channels)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                              diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class Down(nn.Module):
        """Downscaling with maxpool then double conv"""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                UNet.DoubleConv(in_channels, out_channels)
            )

        def forward(self, x):
            return self.maxpool_conv(x)

class patch25_transformer(nn.Module):
    def __init__(self, input_dim, reduce_dim, n_classes, embed_dim, reduce_method):
        super().__init__()

        # input processing and dimensionality reduction
        if reduce_method == 'fixed':
            self.input_processing = FixedReduction(input_dim)
        elif reduce_method == 'linear':
            self.input_processing = LinearReduction(input_dim, reduce_dim)

        self.d_model = embed_dim
        self.cls_token = nn.Parameter(torch.normal(torch.zeros((1, 1, self.d_model)), 0.02))

        # Initial projection into tokens
        self.patch_project = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, 25, 25),
        )

        # Transformer encoder
        self.pe = self.LearnablePositionEncoding(
            d_model=self.d_model,
            dropout=0,
            sequence_length=reduce_dim + 1
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=16,
            dim_feedforward=256,
            activation="gelu",
            batch_first=True)
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=6)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, n_classes),
        )

    class LearnablePositionEncoding(nn.Module):
        def __init__(self, d_model: int, dropout: float = 0.0, sequence_length: int = 5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            self.pe = nn.Parameter(
                torch.tensor(np.random.normal(loc=0.0, scale=0.02, size=(1, sequence_length, d_model)),
                             dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe
            return self.dropout(x)

    def forward(self, x):
        x = self.input_processing(x)

        # Project input 25px patch into a sequence of tokens
        x = self.patch_project(x.unsqueeze(1))
        x = x.squeeze(-1, -2).permute(0, 2, 1)

        # Prepend CLS token to sequence
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        # Add position encodings
        x_pe = self.pe(x)

        # Encoder
        x_tr = self.encoder_stack(x_pe)

        # Classify the CLS token
        logits = self.classifier(x_tr[:, 0])
        return logits

# Inherits from https://docs.pytorch.org/vision/0.12/_modules/torchvision/models/vision_transformer.html
class BlockViT(torchvision.models.VisionTransformer):
    def __init__(self,  input_dim, reduce_dim, n_classes, reduce_method, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # input processing and dimensionality reduction
        if reduce_method == 'fixed':
            self.input_processing = FixedReduction(input_dim)
        elif reduce_method == 'linear':
            self.input_processing = LinearReduction(input_dim, reduce_dim)

        # Projection of each 16x16px patch into a token
        self.conv_proj = nn.Conv2d(
            in_channels=reduce_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Convert tokens back into patches
        self.upscale = nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=16, stride=16)

        # U-Net-like classificiation head to yield segmentation
        self.heads = nn.Sequential(
            nn.Conv2d(self.hidden_dim + reduce_dim,
                      self.hidden_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim // 2, self.hidden_dim // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim // 4, n_classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        hpatch, wpatch = h // self.patch_size, w // self.patch_size

        # Reshape and permute the input tensor
        x_in = self.input_processing(x)
        x = self._process_input(x_in)
        n = x.shape[0]

        # Expand the class token to the full batch
        # Note: this is only included to mesh with torchvision VisionTransformer functions;
        # batch_class_token is never used
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Use token sequence EXCLUDING CLS token
        x = x[:, 1:].permute(0, 2, 1).reshape(b, self.hidden_dim, hpatch, wpatch)

        # Upscale tokens back to 16x16 patches, yielding 256x256 image again
        x = self.upscale(x)

        # Concatenate with input, as in U-Net, then classify
        logits = self.heads(torch.cat([x, x_in], dim=1))

        return logits

