"""Image transforms for training and evaluation."""
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(train, size=224):
    # Only horizontal flip as augmentation. Blur, JPEG recompression and colour
    # jitter would change the pixel-level artifacts that separate GAN images from photos.
    steps = [transforms.Resize((size, size))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)
