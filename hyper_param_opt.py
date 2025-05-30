import numpy as np 
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from typing import Dict, Any
from vae_model_pytorch import AE


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed (seed: int=SEED) -> None: 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
# ---------- I/O helpers ----------
def load_data(filename: str = "data.npz", test_size: float = 0.20, seed: int = 42):
    """
    Loads data and returns (train_val_tensor, test_tensor).
    If 'test_data' exists in the .npz it will be used; otherwise we split.
    """
    data = np.load(filename)
    x = data["train_data"].astype(np.float32)

    if "test_data" in data:
        x_test = data["test_data"].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(x_test)
    else :
        return print("error no test data in data.npz")
    
    
    return x, x_test

def apply_he_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
def objective_train (x, x_test):
    
    def objective(params: Dict[str, Any]):
        lr: float = params["lr"]
        batch_size: int = int(params["batch_size"])
        epochs: int = int(params["epochs"])
        
        # intitiate the model
        model= AE().to(DEVICE)
        
        optimiser=optim.Adam(model.parameters(), lr=lr)
        criterion= nn.MSELoss()
        
        train_loader = DataLoader(x_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(x_test, batch_size=batch_size)
        
        #train the model over epochs and subsequently validate per epoch
        for epoch in range(epochs):
            model.train()
            for x in train_loader:
                x= x.to(DEVICE)
                optimiser.zero_grad()
                out, _= model(x)
                loss= criterion(out,x)
                loss.backward()
                optimiser.step()
            
        model.eval()
        running_loss=0
        with torch.no_grad():
            for (x) in val_loader:
                x=x.to(DEVICE) 
                out, _= model(x)
                loss= criterion(out,x)
                running_loss = loss.item()*x.size(0)
            
        validation_MSE= running_loss/ len(x_test)
        return {"loss": validation_MSE, "status": STATUS_OK, "params": params}
    return objective        
#define search space
search_space = {
    "lr": hp.loguniform("lr", math.log(1e-4), math.log(1e-2)),
    "batch_size": hp.quniform("batch_size", 64, 1024, 64),
    "epochs": hp.quniform("epochs", 50, 1000, 50),
}

if __name__ == "__main__": 
    set_seed()
    
    x_train, x_test = load_data("data.npz")
    objective = objective_train(x_train, x_test)
    
    trials = Trials()
    best= fmin (
        fn=objective,
        space = search_space,
        algo=tpe.suggest,
        max_evals = 50,
        trials= trials
        
    )
    print(best)
    
    
    
    

