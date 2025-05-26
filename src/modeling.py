from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import deepchem as dc
from deepchem.models import GCNModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class DNN(nn.Module):
      
    '''
    Deep Neural Network for binary classification.

    Architecture:
        - Input layer: size input_dim
        - Hidden layer: 64 units with ReLU activation
        - Output layer: 1 unit with sigmoid activation

    Args:
        input_dim (int): Dimension of input features.

    Forward Pass:
        Returns sigmoid-activated output for binary classification.
    '''
       
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class TrainerConvModel:
    
    '''Trainer for conventional machine learning models on fixed-size features.

    Supports:
        - Random Forest
        - Logistic Regression
        - Deep Neural Network

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    each method Returns:
            tuple: (y_pred, y_proba) where:
                y_pred: Binary class predictions (0 or 1)
                y_proba: Probability estimates for positive class
    '''

    def __init__(self, X_train, y_train, X_val, y_val):

            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val

    def train_random_forest(self):
          model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
          model.fit(self.X_train, self.y_train)
          y_pred = model.predict(self.X_val)
          y_proba = model.predict_proba(self.X_val)[:,1]
          return y_pred, y_proba
    
    def train_logistic_regression(self):
          model = LogisticRegression(max_iter=1000, random_state=42)
          model.fit(self.X_train, self.y_train)
          y_pred = model.predict(self.X_val)
          y_proba = model.predict_proba(self.X_val)[:,1]
          return y_pred, y_proba
    
    def train_dnn(self):
          
          X_train_reshaped = np.array(self.X_train).reshape(len(self.y_train), -1)
          X_val_reshaped = np.array(self.X_val).reshape(len(self.y_val), -1)
          train_data = TensorDataset(torch.FloatTensor(X_train_reshaped), torch.FloatTensor(self.y_train))
          train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
          model = DNN(X_train_reshaped.shape[1])
          optimizer =torch.optim.Adam(model.parameters(), lr=0.001)
          criterion = nn.BCELoss()

          for epoch in range(100):
                for x_batch, y_batch in train_loader:
                      optimizer.zero_grad()
                      outputs = model(x_batch).squeeze()
                      loss = criterion(outputs, y_batch)
                      loss.backward()
                      optimizer.step()

          with torch.no_grad():
            y_proba = model(torch.FloatTensor(X_val_reshaped)).squeeze().numpy()
            y_pred = (y_proba > 0.5).astype(int)
          return y_pred, y_proba
    
class TrainerGCN:
     
     '''
     Trainer for Graph Convolutional Network models on molecular graphs.

    Args:
        Graph_train: Training molecular graphs
        Graph_val: Validation molecular graphs

    method train_gcn Returns:
            tuple: (y_pred_class, y_pred) where:
                y_pred_class: Binary class predictions (0 or 1)
                y_pred: Raw prediction scores before thresholding
     '''

     def __init__(self, Graph_train, Graph_val):

            self.Graph_train = Graph_train
            self.Graph_val = Graph_val

     def train_gcn(self):
           model = GCNModel(n_task=1, mode='classification', batch_size=32, learning_rate=0.001)
           model.fit(self.Graph_train, nb_epoch=50)
           y_pred = model.predict(self.Graph_val).flatten()
           y_pred_class = (y_pred > 0.5).astype(int)
           return y_pred_class, y_pred
     
