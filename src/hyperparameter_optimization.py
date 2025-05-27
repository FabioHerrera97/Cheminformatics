import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class LogisticRegressionOptuna:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def objective(self, trial):
        C = trial.suggest_loguniform('C', 1e-4, 10)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])

        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            raise optuna.exceptions.TrialPruned()

        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        proba = model.predict_proba(self.X_val)[:, 1]
        return roc_auc_score(self.y_val, proba)

    def optimize(self, n_trials=50):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        print("Best params:", study.best_params)

        # Train final model
        best_params = study.best_params
        model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        y_proba = model.predict_proba(self.X_val)[:, 1]
        return y_pred, y_proba

class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout):
        super().__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DNNOptuna:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = torch.FloatTensor(np.array(X_train))
        self.y_train = torch.FloatTensor(np.array(y_train))
        self.X_val = torch.FloatTensor(np.array(X_val))
        self.y_val = torch.FloatTensor(np.array(y_val))

    def objective(self, trial):
        hidden_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_sizes = [trial.suggest_int(f"n_units_l{i}", 32, 256) for i in range(hidden_layers)]
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        epochs = 30

        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = DNN(self.X_train.shape[1], hidden_sizes, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = model(x_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_proba = model(self.X_val).squeeze().numpy()
        return roc_auc_score(self.y_val.numpy(), y_proba)

    def optimize(self, n_trials=50):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        print("Best params:", study.best_params)

        # Train final model
        best = study.best_params
        hidden_sizes = [best[f"n_units_l{i}"] for i in range(best["n_layers"])]
        model = DNN(self.X_train.shape[1], hidden_sizes, best["dropout"])
        optimizer = torch.optim.Adam(model.parameters(), lr=best["lr"])
        criterion = nn.BCELoss()
        loader = DataLoader(TensorDataset(self.X_train, self.y_train),
                            batch_size=best["batch_size"], shuffle=True)

        model.train()
        for epoch in range(30):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = model(x_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_proba = model(self.X_val).squeeze().numpy()
            y_pred = (y_proba > 0.5).astype(int)
        return y_pred, y_proba