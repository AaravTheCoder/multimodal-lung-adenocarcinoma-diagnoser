import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("🚀 Loading compressed genomic features...")
df = pd.read_csv("compressed_features.csv")

# Separate features (the 100 PCs) from our target label (has_condition)
X = df.drop(columns=["has_condition"]).values
y = df["has_condition"].values

# 2. Create Train/Validation Split (80% training, 20% testing)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert arrays into PyTorch Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1) # reshape for binary loss
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

# 3. Define the Deep Neural Network Architecture
class GenomicClassifier(nn.Module):
    def __init__(self, input_dim):
        super(GenomicClassifier, self).__init__()
        # Deep feedforward blocks with Dropout to prevent overfitting
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid() # Squashes output to a 0-1 probability range
        )
        
    def forward(self, x):
        return self.network(x)

# Instantiate the model
model = GenomicClassifier(input_dim=100)
criterion = nn.BCELoss() # Binary Cross Entropy Loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

print("\n🏋️ Training the Deep Learning Model...")
epochs = 50
batch_size = 32

for epoch in range(1, epochs + 1):
    model.train()
    
    # Simple batching mechanism
    permutation = torch.randperm(X_train_t.size()[0])
    epoch_loss = 0
    
    for i in range(0, X_train_t.size()[0], batch_size):
        optimizer.zero_grad()
        
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_t[indices], y_train_t[indices]
        
        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # Backward pass (Backpropagation)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    # Evaluate every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_preds_binary = (val_preds >= 0.5).float()
            val_acc = accuracy_score(y_val_t.numpy(), val_preds_binary.numpy()) * 100
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {epoch_loss/X_train_t.size()[0]:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

# 4. Final Evaluation Summary
print("\n🎯 Final Model Evaluation Report:")
model.eval()
with torch.no_grad():
    final_preds = model(X_val_t)
    final_preds_binary = (final_preds >= 0.5).float().numpy()

print(classification_report(y_val, final_preds_binary, target_names=["Stage I/II", "Stage III/IV"]))