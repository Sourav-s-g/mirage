"""Model definitions. Each model file has build(img_size) returning a module (B,3,H,W) -> (B,) logits."""
import warnings


def create_timm_model(timm_name, pretrained, **kwargs):
    """timm model with a one-logit head. Falls back to random weights, with a warning, if the download fails."""
    import timm
    if pretrained:
        try:
            # num_classes=1 replaces the 1000-class head with a new single-logit layer
            return timm.create_model(timm_name, pretrained=True, num_classes=1, **kwargs)
        except Exception as e:  # download errors come in many types
            warnings.warn(f"could not load pretrained weights for {timm_name} ({e}). "
                          "Using RANDOM initialisation, results are not comparable to a "
                          "pretrained run.")
    return timm.create_model(timm_name, pretrained=False, num_classes=1, **kwargs)


def get_model(name, img_size=224, pretrained=True):
    # imports are inside the branches so one model's dependencies are only needed when it is used
    if name == "cnn":
        from .cnn import build
        return build(img_size, pretrained, arch="xception")
    elif name == "cnn_resnet34":
        from .cnn import build
        return build(img_size, pretrained, arch="resnet34")
    elif name == "lstm":
        from .lstm import build
        return build(img_size=img_size)
    elif name == "vit":
        from .vit import build
        return build(img_size, pretrained, variant="small")
    elif name == "vit_base":
        from .vit import build
        return build(img_size, pretrained, variant="base")
    else:
        raise ValueError(f"unknown model: {name}")
