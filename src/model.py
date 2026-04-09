import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np

class FederatedNN(nn.Module):
    def __init__(self, input_dim):
        super(FederatedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x

def train_model(model, dataloader, epochs=5, lr=0.005, sample_weights=None):
    """
    Local training function for a single hospital/bank.
    """
    criterion_unweighted = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for __ in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Simple unweighted training
            loss = criterion_unweighted(outputs, targets)
            loss.backward()
            optimizer.step()

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the given test set.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        targets = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        
        outputs = model(inputs)
        preds = (outputs >= 0.5).int()
        
        acc = accuracy_score(targets.numpy(), preds.numpy())
        
        return acc, preds.numpy().flatten(), outputs.numpy().flatten()
