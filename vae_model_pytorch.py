import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # Encoder: from 20 to 10 to 4 to 2 nodes
        self.encoder = nn.Sequential(
            nn.Linear(20, 10),
            #nn.BatchNorm1d(10),  
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(10, 4),
            #nn.BatchNorm1d(4),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(4, 2),
        )
        # Decoder: from 2 to 4 to 10 to 20 nodes
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            #nn.BatchNorm1d(4),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(4, 10),
            #nn.BatchNorm1d(10),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(10, 20)  
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

if __name__ == "__main__":
    # Print model summary 
    model = AE()
    print(model)


