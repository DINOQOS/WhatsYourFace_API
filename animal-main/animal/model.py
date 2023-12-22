import torch.nn as nn
import torchvision.models as models

class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        
        # Use pre-trained model
        self.model = models.densenet161(pretrained=False)
        
        # Freeze all layers (No training)
        for param in self.parameters():
            param.requires_grad = False
            
        # Change final FC layer to num_classes output. This is trainable by default
        self.model.classifier = nn.Linear(2208, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x
