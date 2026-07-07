"""CNN encoder: pretrained InceptionV3 with the classification head removed.
Early layers stay frozen (ImageNet features are already good general-purpose
edge/texture/shape detectors); the last Inception block is left trainable so
the network can adapt its higher-level visual features to the captioning
task and the specific look of the training images."""
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

IMAGE_SIZE = 299  # InceptionV3's expected input size
UNFREEZE_FROM = "Mixed_7a"  # first of the final 3 Inception blocks

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, unfreeze_from=UNFREEZE_FROM):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        inception.fc = nn.Identity()
        inception.aux_logits = False
        self.inception = inception
        self._set_trainable_layers(unfreeze_from)

        self.linear = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def _set_trainable_layers(self, unfreeze_from):
        trainable = unfreeze_from is None
        for name, child in self.inception.named_children():
            if name == unfreeze_from:
                trainable = True
            for param in child.parameters():
                param.requires_grad = trainable

    def forward(self, images):
        features = self.inception(images)
        return self.bn(self.linear(features))
