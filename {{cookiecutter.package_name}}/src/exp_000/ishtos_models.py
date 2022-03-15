import timm
import torch
import torch.nn as nn


class HeadV1(nn.Module):
    def __init__(self, in_features, out_features):
        super(HeadV1, self).__init__()
        self.head = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        return self.head(x)


# --------------------------------------------------
# ResNet
# - resnet18, 26, 34, 50, 101, 152, 200
# - resnet18d, 26, 34, 50, 101, 152, 200
# --------------------------------------------------
class ResNet(nn.Module):
    def __init__(
        self,
        base_model="resnet18",
        pretrained=True,
        my_pretrained=None,
        num_classes=1,
        head_version="v1",
    ):
        super(ResNet, self).__init__()
        self.model = timm.create_model(base_model, pretrained=pretrained)
        if my_pretrained:
            self.model.load_state_dict(torch.load(my_pretrained))
        in_features = self.model.fc.in_features
        self.model.fc = get_head(
            version=head_version, in_features=in_features, out_features=num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x


# --------------------------------------------------
# ConvNeXt
# - convnext_tiny, small, base, large
# - convnext_base_in22ft1k, large, xlarge
# - convnext_base_384_in22ft1k, large, xlarge
# - convnext_base_in22k, large, xlarge
# --------------------------------------------------
class ConvNeXt(nn.Module):
    def __init__(
        self,
        base_model="convnext_base_in22k",
        pretrained=True,
        my_pretrained=None,
        num_classes=1,
        head_version="v1",
    ):
        super(ConvNeXt, self).__init__()

        self.model = timm.create_model(base_model, pretrained=pretrained)
        if my_pretrained:
            self.model.load_state_dict(torch.load(my_pretrained))

        in_features = self.model.head.fc.in_features
        self.model.head.fc = get_head(
            version=head_version, in_features=in_features, out_features=num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x


# --------------------------------------------------
# EfficientNet
# - efficientnet_b0 ~ b4
# - efficientnet_es, m, l
# - efficientnet_es_pruned, l
# - efficientnet_b1_pruned ~ b3
# - efficientnetv2_rw_t, s, m
# - tf_efficientnet_b0 ~ b8
# - tf_efficientnet_b0_ap ~ b8
# - tf_efficientnet_b0_ns ~ b7
# - tf-efficientnet_es, m, l
# - tf_efficientnetv2_s, m, l
# - tf_efficientnetv2_s_in21k, m, l, xl
# - tf_efficientnetv2_b0 ~ b3
# --------------------------------------------------
class EfficientNet(nn.Module):
    def __init__(
        self,
        base_model="efficientnet_b0",
        pretrained=True,
        my_pretrained=None,
        num_classes=1,
        head_version="v1",
    ):
        super(EfficientNet, self).__init__()

        self.model = timm.create_model(base_model, pretrained=pretrained)
        if my_pretrained:
            self.model.load_state_dict(torch.load(my_pretrained))

        in_features = self.model.classifier.in_features
        self.model.classifier = get_head(
            version=head_version, in_features=in_features, out_features=num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x


# --------------------------------------------------
# SwinTransformer
# - swin_base_patch4_window12_384, large
# - swin_base_patch4_window7_224, tiny, small, large
# - swin_base_patch4_window12_384_in22k, large
# - swin_base_patch4_window7_224_in22k, large
# --------------------------------------------------
class SwinTransformer(nn.Module):
    def __init__(
        self,
        base_model="swin_tiny_patch4_window7_224",
        pretrained=True,
        my_pretrained=None,
        num_classes=1,
        head_version="v1",
    ):
        super(SwinTransformer, self).__init__()

        self.model = timm.create_model(base_model, pretrained=pretrained)
        if my_pretrained:
            self.model.load_state_dict(torch.load(my_pretrained))

        in_features = self.model.head.in_features
        self.model.head = get_head(
            version=head_version, in_features=in_features, out_features=num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_model(config):
    model_name = config.name
    if model_name == "convnext":
        return ConvNeXt(**config.params)
    elif model_name == "efficientnet":
        return EfficientNet(**config.params)
    elif model_name == "resnet":
        return ResNet(**config.params)
    elif model_name == "swin":
        return SwinTransformer(**config.params)
    else:
        raise ValueError(f"Not supported model: {model_name}")


def get_head(version, in_features, out_features):
    if version == "v1":
        return HeadV1(in_features=in_features, out_features=out_features)
    else:
        raise ValueError(f"Not supported head version: {version}")
