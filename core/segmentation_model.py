import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ResNetUNet, self).__init__()

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.pool1 = self.resnet.maxpool
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder5 = self._make_decoder_block(512, 256)

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = self._make_decoder_block(256, 128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = self._make_decoder_block(128, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = self._make_decoder_block(64, 32)

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = self._make_decoder_block(16, 16)

        self.conv_final = nn.Conv2d(16, out_channels, kernel_size=1)

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d5 = self.upconv5(e5)
        d5 = torch.cat((d5, e4), dim=1)
        d5 = self.decoder5(d5)

        d4 = self.upconv4(d5)
        d4 = torch.cat((d4, e3), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = self.decoder1(d1)

        out = self.conv_final(d1)
        return torch.sigmoid(out)


class LegacySimpleResNetUNet(nn.Module):
    """
    Compatibility model for checkpoints that store keys like
    conv1/bn1/layer1-4 and decoder upconv4..0/final_conv without the
    nested `resnet.` prefix. This mirrors the shape layout found in
    simple_resnet_unet_best.pth.
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.relu(self.upconv4(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv0(x))
        x = self.final_conv(x)
        return torch.sigmoid(x)


def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def combined_loss(pred, target, bce_weight=0.5, dice_weight=0.5):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dice


def load_segmentation_model(checkpoint_path, device):
    """
    Load segmentation model handling both the current ResNetUNet and the
    legacy simple_resnet_unet checkpoints (with keys conv1/bn1/layer*/upconv*/final_conv).
    """
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    # Legacy format: top-level conv1/layer1... and upconv*/final_conv keys
    legacy_keys = {'conv1.weight', 'upconv4.weight', 'final_conv.weight'}
    if legacy_keys.issubset(set(state.keys())):
        model = LegacySimpleResNetUNet()
        model.load_state_dict(state)
    else:
        model = ResNetUNet()
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model
