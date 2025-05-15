import torch.nn as nn
import torch.nn.functional as F

class ImprovedCNN(nn.Module):
    """
    CNN architecture for CIFAR-10 image classification.
    
    This model consists of several convolutional layers with batch normalization,
    max pooling, dropout for regularization, and fully connected layers.
    """
    def __init__(self):
        super().__init__()
        
        # First convolutional layer: input 3 channels (RGB), output 32 filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer: input 32, output 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.dropout1 = nn.Dropout(0.25)

        # Third convolutional layer: input 64, output 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fourth convolutional layer: maintains output at 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer: input 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            Output tensor with predictions for each class
        """
        # First block: conv -> batch norm -> relu -> pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second block: conv -> batch norm -> relu -> pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        
        # Third block: conv -> batch norm -> relu
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Fourth block: conv -> batch norm -> relu -> pooling
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten tensor for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x 