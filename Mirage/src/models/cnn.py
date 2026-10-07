"""ImageNet-pretrained CNN with a one-logit head. Xception is the main model, ResNet-34 the second."""
from torch import nn

from . import create_timm_model

# For each architecture: the timm name with its weight tag (so the weights do not change with the
# timm version) and the layer holding the last feature map before pooling, the Grad-CAM target.
ARCHS = {
    "xception": {"timm_name": "legacy_xception.tf_in1k", "cam_layer": "act4"},
    "resnet34": {"timm_name": "resnet34.tv_in1k", "cam_layer": "layer4"},
}

# the pretrained backbone is changed gently, the new head has to learn from scratch
BACKBONE_LR_FACTOR = 0.1


class CNN(nn.Module):
    def __init__(self, timm_name, pretrained):
        super().__init__()
        self.net = create_timm_model(timm_name, pretrained)

    def forward(self, x):
        return self.net(x).squeeze(1)

    def param_groups(self, lr):
        """Backbone at a lower learning rate than the head."""
        head = list(self.net.get_classifier().parameters())
        head_ids = {id(p) for p in head}
        backbone = [p for p in self.net.parameters() if id(p) not in head_ids]
        return [{"params": backbone, "lr": lr * BACKBONE_LR_FACTOR},
                {"params": head, "lr": lr}]


def build(img_size=224, pretrained=True, arch="xception"):
    # img_size is unused: the network is fully convolutional with global pooling
    return CNN(ARCHS[arch]["timm_name"], pretrained)
