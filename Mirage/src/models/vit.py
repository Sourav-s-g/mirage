"""ImageNet-pretrained Vision Transformer with a one-logit head. Small is the default, base is selectable."""
from torch import nn

from . import create_timm_model

# timm names with the weight tag, so the weights do not change with the timm version
VARIANTS = {
    "small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
}

# the learning rate rises linearly over this fraction of all training steps, then follows the cosine
WARMUP_FRACTION = 0.1


class ViT(nn.Module):
    def __init__(self, timm_name, pretrained, img_size):
        super().__init__()
        # img_size other than 224 makes timm resize the pretrained position embeddings
        self.net = create_timm_model(timm_name, pretrained, img_size=img_size)
        self.warmup_fraction = WARMUP_FRACTION

    def forward(self, x):
        return self.net(x).squeeze(1)


def build(img_size=224, pretrained=True, variant="small"):
    return ViT(VARIANTS[variant], pretrained, img_size)


def cam_target_layer(model):
    """Layer for Grad-CAM: the first norm of the last transformer block, outputs (B, 1 + N, C) tokens."""
    return model.net.blocks[-1].norm1


def reshape_transform(tokens):
    """(B, 1 + N, C) tokens -> (B, C, h, w) grid, with h = w = sqrt(N) (14 for a 224 px image).

    Grad-CAM needs activations laid out as a 2D map, so that each position can be weighted and the
    result drawn as a heatmap over the image. A ViT layer outputs a flat token sequence instead.
    The first token is the class token, which belongs to no image position, so it is dropped. The
    patch tokens are in row-major order, so reshaping them to (h, w) restores their positions.
    """
    patches = tokens[:, 1:, :]
    side = int(patches.shape[1] ** 0.5)
    grid = patches.reshape(patches.shape[0], side, side, patches.shape[2])
    return grid.permute(0, 3, 1, 2)
