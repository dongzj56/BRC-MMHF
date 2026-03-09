from torch import nn
import torch
import time

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv3DBlock(nn.Module):
    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))

    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out


class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    def forward(self, input):
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        return out

class UNet3D_Feature(nn.Module):
    def __init__(self,
                 in_channels: int,
                 level_channels: list = [64, 128, 256],
                 bottleneck_channel: int = 512) -> None:
        super(UNet3D_Feature, self).__init__()
        l1, l2, l3 = level_channels

        self.a_block1 = Conv3DBlock(in_channels, l1)
        self.a_block2 = Conv3DBlock(l1, l2)
        self.a_block3 = Conv3DBlock(l2, l3)
        self.bottleNeck = Conv3DBlock(l3, bottleneck_channel, bottleneck=True)

        self.s_block3 = UpConv3DBlock(bottleneck_channel, res_channels=l3)
        self.s_block2 = UpConv3DBlock(l3, res_channels=l2)
        self.s_block1 = UpConv3DBlock(l2, res_channels=l1,
                                      last_layer=False, num_classes=None)

    def forward(self, x):
        x, res1 = self.a_block1(x)
        x, res2 = self.a_block2(x)
        x, res3 = self.a_block3(x)
        x, _    = self.bottleNeck(x)

        x = self.s_block3(x, res3)
        x = self.s_block2(x, res2)
        x = self.s_block1(x, res1)
        return x

class UNet3DClassifier(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 level_channels=[64, 128, 256],
                 bottleneck_channel=512):
        super(UNet3DClassifier, self).__init__()
        l1, l2, l3 = level_channels

        self.a_block1    = Conv3DBlock(in_channels,   l1)
        self.a_block2    = Conv3DBlock(l1,            l2)
        self.a_block3    = Conv3DBlock(l2,            l3)
        self.bottleNeck  = Conv3DBlock(l3, bottleneck_channel, bottleneck=True)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier  = nn.Linear(bottleneck_channel, num_classes)

    def forward(self, x):
        x, _ = self.a_block1(x)
        x, _ = self.a_block2(x)
        x, _ = self.a_block3(x)
        x, _ = self.bottleNeck(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


class UNet3DEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 level_channels = [64, 128, 256],
                 bottleneck_channel: int = 512):
        super().__init__()
        l1, l2, l3 = level_channels
        self.block1      = Conv3DBlock(in_channels, l1)
        self.block2      = Conv3DBlock(l1, l2)
        self.block3      = Conv3DBlock(l2, l3)
        self.bottleneck  = Conv3DBlock(l3, bottleneck_channel, bottleneck=True)

    def forward(self, x):
        x, _ = self.block1(x)
        x, _ = self.block2(x)
        x, _ = self.block3(x)
        x, _ = self.bottleneck(x)
        return x

class DualStreamUNet3DClassifier(nn.Module):
    def __init__(self,
                 in_channels_per_modality: int = 1,
                 num_classes: int = 2,
                 level_channels = [64, 128, 256],
                 bottleneck_channel: int = 512):
        super().__init__()
        self.mri_enc = UNet3DEncoder(in_channels_per_modality,
                                     level_channels,
                                     bottleneck_channel)
        self.pet_enc = UNet3DEncoder(in_channels_per_modality,
                                     level_channels,
                                     bottleneck_channel)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        fused_dim = bottleneck_channel * 2
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, mri, pet):
        f_mri = self.mri_enc(mri)
        f_pet = self.pet_enc(pet)

        v_mri = self.global_pool(f_mri).view(f_mri.size(0), -1)
        v_pet = self.global_pool(f_pet).view(f_pet.size(0), -1)

        fused = torch.cat([v_mri, v_pet], dim=1)
        logits = self.classifier(fused)
        return logits

def cen_exchange(x_mri, x_pet, half_ratio=0.5):
    B, C, D, H, W = x_mri.shape
    k = int(C * half_ratio)
    if k == 0:
        return x_mri, x_pet
    xm_head, xm_tail = x_mri[:, :k], x_mri[:, k:]
    xp_head, xp_tail = x_pet[:, :k], x_pet[:, k:]
    x_mri_new = torch.cat([xp_head, xm_tail], dim=1)
    x_pet_new = torch.cat([xm_head, xp_tail], dim=1)
    return x_mri_new, x_pet_new


class SharedConv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, half_ratio=0.5, with_pool=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_ch // 2, out_ch, kernel_size=3, padding=1)
        self.bn1_mri = nn.BatchNorm3d(out_ch // 2)
        self.bn1_pet = nn.BatchNorm3d(out_ch // 2)
        self.bn2_mri = nn.BatchNorm3d(out_ch)
        self.bn2_pet = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.with_pool = with_pool
        if with_pool:
            self.pool = nn.MaxPool3d(2, 2)
        self.half_ratio = half_ratio

    def _forward_branch(self, x, bn1, bn2):
        x = self.relu(bn1(self.conv1(x)))
        x = self.relu(bn2(self.conv2(x)))
        return x

    def forward(self, x_mri, x_pet):
        xm = self._forward_branch(x_mri, self.bn1_mri, self.bn2_mri)
        xp = self._forward_branch(x_pet, self.bn1_pet, self.bn2_pet)
        xm, xp = cen_exchange(xm, xp, self.half_ratio)
        if self.with_pool:
            xm_p = self.pool(xm)
            xp_p = self.pool(xp)
            return xm_p, xp_p, xm, xp
        else:
            return xm, xp, xm, xp

class PartialCENUNet3DClassifier(nn.Module):
    def __init__(self,
                 in_ch_modality   = 1,
                 num_classes      = 2,
                 level_channels   = [64, 128, 256],
                 bottleneck_ch    = 512,
                 share_layers     = 2,
                 cen_ratios       = (0.2, 0.1)):
        super().__init__()
        assert share_layers == len(cen_ratios), \
            "len(cen_ratios) must equal share_layers"

        l1, l2, l3 = level_channels

        self.shared_blocks = nn.ModuleList()
        in_out_pairs = [(in_ch_modality, l1),
                        (l1, l2),
                        (l2, l3)]
        for i in range(share_layers):
            in_c , out_c  = in_out_pairs[i]
            ratio         = cen_ratios[i]
            self.shared_blocks.append(
                SharedConv3DBlock(in_c, out_c, half_ratio=ratio)
            )

        last_out = [l1, l2, l3][share_layers-1]

        self.mri_block3 = Conv3DBlock(last_out, l3)
        self.pet_block3 = Conv3DBlock(last_out, l3)
        self.mri_bneck  = Conv3DBlock(l3, bottleneck_ch, bottleneck=True)
        self.pet_bneck  = Conv3DBlock(l3, bottleneck_ch, bottleneck=True)

        self.gap        = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(bottleneck_ch * 2, num_classes)

    def forward(self, mri, pet):
        for blk in self.shared_blocks:
            mri, pet, _, _ = blk(mri, pet)

        mri, _ = self.mri_block3(mri)
        pet, _ = self.pet_block3(pet)
        mri, _ = self.mri_bneck(mri)
        pet, _ = self.pet_bneck(pet)

        vm = self.gap(mri).flatten(1)
        vp = self.gap(pet).flatten(1)
        fused  = torch.cat([vm, vp], dim=1)
        logits = self.classifier(fused)
        return logits


if __name__ == '__main__':
    device = torch.device("cpu")
    model = UNet3D(in_channels=2,
                                       num_classes=2).to(device)

    B, C, D, H, W = 1, 2, 96, 112, 96
    mri = torch.randn(B, C, D, H, W, device=device)
    pet = torch.randn(B, C, D, H, W, device=device)

    start = time.time()
    out = model(mri)
    print("logits shape:", out.shape)
    summary(model, [(C, D, H, W)], device="cpu")
    print("elapsed:", time.time() - start, "s")
