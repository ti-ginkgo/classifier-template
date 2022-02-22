import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms_v1(config, pretrained):
    augmentations = [A.Resize(config.height, config.width)]
    if pretrained:
        augmentations.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        augmentations.append(A.Normalize(mean=0, std=1))
    augmentations.append(ToTensorV2())
    return A.Compose(augmentations)


def get_valid_transforms_v1(config, pretrained):
    augmentations = [A.Resize(config.height, config.width)]
    if pretrained:
        augmentations.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        augmentations.append(A.Normalize(mean=0, std=1))
    augmentations.append(ToTensorV2())
    return A.Compose(augmentations)


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_transforms(config, phase):
    try:
        if phase == "train":
            return eval(f"get_train_transforms_{config.transforms.train_version}")(
                config.transforms.params, config.model.params.pretrained,
            )
        elif phase in ["valid", "test"]:
            return eval(f"get_valid_transforms_{config.transforms.valid_version}")(
                config.transforms.params, config.model.params.pretrained,
            )
        else:
            raise ValueError(f"Not supported transforms phase: {phase}.")
    except NameError:
        return None
