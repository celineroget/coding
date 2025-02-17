
# import libraries
import numpy as np
import random
# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class mlp(nn.Module):

  def __init__(self,
               time_periods, n_classes):
        super(mlp, self).__init__()
        self.time_periods = time_periods
        self.n_classes = n_classes
        # WRITE CODE HERE
        self.flatten = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(3*self.time_periods, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        
#        raise NotImplementedError

  def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = nn.Flatten()(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x
  
class cnn(nn.Module):

  def __init__(self, time_periods, n_sensors, n_classes):
        super(cnn, self).__init__()
        self.time_periods = time_periods
        self.n_sensors = n_sensors
        self.n_classes = n_classes

        # WRITE CODE HERE
        # Convolutional layers
        self.conv1 = nn.Conv1d(self.n_sensors, 100, 10, padding=0, stride=1)
        self.conv2 = nn.Conv1d(100, 100, 10, padding=0, stride=1)
        self.conv3 = nn.Conv1d(100, 160, 10, padding=0, stride=1)
        self.conv4 = nn.Conv1d(160, 160, 10, padding=0, stride=1)
        # Pooling and dropout
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.dropout = nn.Dropout(0.5)
        # Adaptive pool layer to adjust the size before sending to fully connected layer
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # Fully connected layer
        self.fc = nn.Linear(160, self.n_classes)
       
        

  def forward(self, x):
        # Reshape the input to (batch_size, n_sensors, time_periods)
        # WRITE CODE HERE
        
        # Layers
        # WRITE CODE HERE
       # Reshape the input to (batch_size, n_sensors, time_periods)
        x = x.view(x.size(0),self.n_sensors, self.time_periods)
#       # Convolutional layers with ReLU activations

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
       

        # Global average pooling and dropout
        x = self.global_avg_pool(x)
        
        x = self.dropout(x)
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 160) 
        x = self.fc(x)
        # Output layer with softmax activation
        x = F.log_softmax(x, dim=1)
        return x
         
