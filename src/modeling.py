from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib

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
    
    '''A class for training and evaluating machine learning models with combined training/validation data.
    
    This class combines training and validation datasets to maximize training data usage, then trains
    both logistic regression and deep neural network (DNN) models using pre-optimized hyperparameters.
    Finally, it evaluates performance on a held-out test set and saves the trained models.

    Args:
        X_train (array-like): Training feature matrix
        y_train (array-like): Training target vector
        X_val (array-like): Validation feature matrix
        y_val (array-like): Validation target vector
        X_test (array-like): Test feature matrix
        y_test (array-like): Test target vector

    Attributes:
        X_train (ndarray): Combined training+validation features
        y_train (ndarray): Combined training+validation targets
        X_test (ndarray): Test features
        y_test (ndarray): Test targets
        lr_params (dict): Optimal parameters for logistic regression:
            - C (float): Inverse regularization strength (2.002)
            - penalty (str): Regularization type ('l2')
            - solver (str): Optimization algorithm ('liblinear')
            - max_iter (int): Maximum iterations (1000)
            - random_state (int): Random seed (42)
        dnn_params (dict): Optimal parameters for DNN:
            - n_layers (int): Number of hidden layers (2)
            - n_units (list): Units per layer [112, 144]
            - dropout (float): Dropout probability (0.351)
            - lr (float): Learning rate (0.00133)
            - batch_size (int): Training batch size (32)

    Methods:
        train_logistic_regression():
            Trains logistic regression model with optimal hyperparameters.
            
        train_dnn():
            Trains deep neural network with optimal architecture and hyperparameters.
    '''
     
class TrainOptimalModels:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = np.concatenate([X_train, X_val])
        self.y_train = np.concatenate([y_train, y_val])
        self.X_test = X_test
        self.y_test = y_test
        
        self.lr_params = {
            'C': 2.0021859860317472,
            'penalty': 'l2',
            'solver': 'liblinear',
            'max_iter': 1000,
            'random_state': 42
        }
        
        self.dnn_params = {
            'n_layers': 2,
            'n_units': [112, 144], 
            'dropout': 0.35104406840256835,
            'lr': 0.0013305808434076489,
            'batch_size': 32
        }
    
    def train_logistic_regression(self):
        """Trains logistic regression with optimal hyperparameters."""
        model = LogisticRegression(**self.lr_params)
        model.fit(self.X_train, self.y_train)
        
        y_test_pred = model.predict(self.X_test)
        y_test_proba = model.predict_proba(self.X_test)[:,1]
        
        joblib.dump(model, 'logistic_regression_model.pkl')
        return y_test_pred, y_test_proba
    
    def train_dnn(self):
        """Trains DNN with optimal architecture and hyperparameters."""
        class DNN(nn.Module):
            def __init__(self, input_size, dnn_params):
                super(DNN, self).__init__()
                self.layers = nn.ModuleList()
                
                self.layers.append(nn.Linear(input_size, dnn_params['n_units'][0]))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dnn_params['dropout']))
                
                for i in range(1, dnn_params['n_layers']):
                    self.layers.append(nn.Linear(dnn_params['n_units'][i-1], dnn_params['n_units'][i]))
                    self.layers.append(nn.ReLU())
                    self.layers.append(nn.Dropout(dnn_params['dropout']))
                
                self.layers.append(nn.Linear(dnn_params['n_units'][-1], 1))
                self.layers.append(nn.Sigmoid())
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        X_reshaped = np.array(self.X_train).reshape(len(self.y_train), -1)
        test_reshaped = np.array(self.X_test).reshape(len(self.y_test), -1)
        
        train_data = TensorDataset(torch.FloatTensor(X_reshaped), torch.FloatTensor(self.y_train))
        train_loader = DataLoader(train_data, batch_size=self.dnn_params['batch_size'], shuffle=True)
        
        model = DNN(X_reshaped.shape[1], self.dnn_params)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.dnn_params['lr'])
        criterion = nn.BCELoss()
        
        for epoch in range(100):
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(x_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            y_test_proba = model(torch.FloatTensor(test_reshaped)).squeeze().numpy()
            y_test_pred = (y_test_proba > 0.5).astype(int)
        
        torch.save(model.state_dict(), 'dnn_model.pth')
        return y_test_pred, y_test_proba