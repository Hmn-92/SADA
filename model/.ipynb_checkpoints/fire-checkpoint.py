"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

import copy

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from timm import create_model
import timm

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class Classifier(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=-1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        return self.classifier(x)


class FgClassifier(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=-1, init_center=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.weight = nn.Parameter(copy.deepcopy(init_center))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x_norm, w)


class AttrAwareLoss(nn.Module):
    def __init__(self, scale=16, epsilon=0.1):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, positive_mask):
        inputs = self.scale * inputs
        identity_mask = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()

        log_probs = self.logsoftmax(inputs)
        mask = (1 - self.epsilon) * identity_mask + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (- mask * log_probs).mean(0).sum()
        return loss


class MaxAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_f = self.maxpooling(x)
        avg_f = self.avgpooling(x)
        return torch.cat((max_f, avg_f), 1)


class MultiGranularFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=2048, P_parts=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.P_parts = P_parts
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 动态权重的注意力层，分别针对上半身和下半身
        self.attn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim // 2, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(self.P_parts)
        ])

    def forward(self, x):
        # 全局特征池化
        global_feat = self.global_pool(x).view(x.size(0), -1)  # [B, feature_dim]

        # 上半身和下半身池化
        h = x.size(2) // self.P_parts
        upper_feat = x[:, :, :h, :].mean(dim=[2, 3])  # [B, feature_dim]
        lower_feat = x[:, :, h:, :].mean(dim=[2, 3])  # [B, feature_dim]

        # 将特征进行拼接，确保拼接后的维度正确
        multi_granular_feat = torch.cat([global_feat, upper_feat, lower_feat], dim=1)  # [B, feature_dim * 3]
        return multi_granular_feat


class AttentionFeatureRecomposition(nn.Module):
    def __init__(self, feature_dim=2048, P_parts=2, K_times=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.P_parts = P_parts
        self.K_times = K_times
        # 自注意力模块
        self.attention_layer = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),  # 保持通道数不变
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),  # 保持通道数不变
            nn.Sigmoid()
        )
        # 全局池化，用于得到最终特征
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features, fgid):
        B, D, H, W = features.size()  # [B, D, H, W]

        if self.training and fgid is not None:
            part_h = features.shape[2] // self.P_parts
            FAR_parts = []

            # 对每个部分应用注意力
            for k in range(self.P_parts):
                part = features[:, :, part_h * k: part_h * (k + 1), :]  # [B, D, h', w]

                # 添加注意力层
                attn_weight = self.attention_layer(part)  # [B, D, h', w]
                part = part * attn_weight  # 应用注意力权重增强特征

                # 均值和方差计算
                mu = part.mean(dim=[2, 3], keepdim=True)
                var = part.var(dim=[2, 3], keepdim=True)
                sig = (var + 1e-6).sqrt()
                mu, sig = mu.detach(), sig.detach()  # [B, D, 1, 1]
                id_part = (part - mu) / sig  # [B, D, h', w]

                # 负样本采样
                neg_mask = fgid.expand(B, B).ne(fgid.expand(B, B).t())  # [B, B]
                neg_mask = neg_mask.type(torch.float32)
                sampled_idx = torch.multinomial(neg_mask, num_samples=self.K_times, replacement=False).transpose(-1, -2).flatten(0)  # [B, K] -> [BK]
                new_mu = mu[sampled_idx]  # [BK, D, 1, 1]
                new_sig = sig[sampled_idx]  # [BK, D, 1, 1]

                # 重组特征
                id_part = id_part.repeat(self.K_times, 1, 1, 1)
                FAR_part = (id_part * new_sig) + new_mu  # [B, D, h', w]
                FAR_parts.append(FAR_part)

            # 拼接所有重组后的部分特征
            FAR_feat = torch.concat(FAR_parts, dim=2)  # [B, D, h, w]
            FAR_feat = self.global_pool(FAR_feat).flatten(1)  # [B, D]

            return FAR_feat




class FIRe(nn.Module):
    def __init__(self, pool_type='avg', last_stride=1, pretrain=True, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.P_parts = 2
        self.K_times = 1

        resnet = getattr(torchvision.models, 'resnet101')(pretrained=pretrain)
        resnet.layer4[0].downsample[0].stride = (last_stride, last_stride)
        resnet.layer4[0].conv2.stride = (last_stride, last_stride)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # feature_dim = 2048


        feature_dim = 2048 # ResNet + Transformer 的特征维度

        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'maxavg':
            self.pool = MaxAvgPool2d()
        self.feature_dim = (3 * feature_dim) if pool_type == 'maxavg' else feature_dim

        # 修改bottleneck，确保维度匹配
        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.FAR_bottleneck = nn.BatchNorm1d(feature_dim)
        self.FAR_bottleneck.bias.requires_grad_(False)
        self.FAR_bottleneck.apply(weights_init_kaiming)
        self.FAR_classifier = nn.Linear(feature_dim, self.num_classes, bias=False)
        self.FAR_classifier.apply(weights_init_classifier)

        # 新增模块
        self.multi_granular_extractor = MultiGranularFeatureExtractor(feature_dim, P_parts=self.P_parts)
        self.attn_recomposer = AttentionFeatureRecomposition(feature_dim, P_parts=self.P_parts, K_times=self.K_times)

    def forward(self, x, fgid=None):
        B = x.shape[0]
        features = self.backbone(x)
        # 提取多粒度特征
        multi_granular_feat = self.multi_granular_extractor(features)
        global_feat_bn = self.bottleneck(multi_granular_feat)

        # 在训练时进行动态重组
        if self.training and fgid is not None:
            FAR_feat = self.attn_recomposer(features, fgid)
            FAR_feat_bn = self.FAR_bottleneck(FAR_feat)
            y_FAR = self.FAR_classifier(FAR_feat_bn)
            return global_feat_bn, y_FAR

        else:
            return global_feat_bn