import torch
from torch import nn
from torchvision import models, transforms
from .catalog import TASKS


def preprocessing():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


class CytologyModel(nn.Module):
    def __init__(self, architecture, pretrained=False):
        super().__init__()
        self.architecture = architecture
        if architecture == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.target_layer = backbone.layer4
            size = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif architecture == 'efficientnet_b0':
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.target_layer = backbone.features[-1]
            size = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError('Unsupported architecture')
        self.backbone = backbone
        self.heads = nn.ModuleDict({task: nn.Linear(size, len(classes)) for task, classes in TASKS.items()})

    def forward(self, x, task):
        return self.heads[task](self.backbone(x))


def gradcam(model, tensor, task, class_index):
    activations, gradients = [], []
    def capture(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))
    hook = model.target_layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(tensor.detach().requires_grad_(True), task)
            logits[0, class_index].backward()
            cam = (gradients[0].mean((2, 3), keepdim=True) * activations[0]).sum(1).relu()
            cam = torch.nn.functional.interpolate(cam[:, None], (224, 224), mode='bilinear', align_corners=False)[0, 0]
            cam = cam.detach().cpu()
            return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).numpy()
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
