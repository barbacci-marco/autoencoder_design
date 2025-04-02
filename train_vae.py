import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vae_model_pytorch import AE

def load_data(filename='data.npz'):
  
    data = np.load(filename)
    train_data = data['train_data'].astype(np.float32)
    return train_data

def apply_he_init(model):
   
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()

def train_autoencoder(epochs=100, batch_size=32, learning_rate=1e-3, l1_reg=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE().to(device)
    
    # Apply He initialization to the model
    apply_he_init(model)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Load training data and create a DataLoader with random sampling
    train_data = load_data('data.npz')
    tensor_data = torch.from_numpy(train_data)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch in dataloader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            
            outputs, encoded = model(inputs)
            mse_loss = criterion(outputs, inputs)
            l1_loss = l1_reg * torch.norm(encoded, 1)
            loss = mse_loss + l1_loss
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "autoencoder_weights.pt")
            print(f"Saved best model with loss: {best_loss:.6f}")
    
    return model

if __name__ == "__main__":
    trained_model = train_autoencoder(epochs=10000, batch_size=200, learning_rate=1e-3, l1_reg=1e-5)

