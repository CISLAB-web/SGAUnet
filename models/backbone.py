import segmentation_models_pytorch as smp

def get_encoder(encoder_name="resnet18", pretrained=True):
    encoder = smp.encoders.get_encoder(
        name=encoder_name,
        in_channels=3,
        weights="imagenet" if pretrained else None
    )
    return encoder