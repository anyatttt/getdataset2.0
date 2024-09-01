import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Load the dataset
file_path = "mTORcanonical.csv"  # Update this with your dataset
df = pd.read_csv(file_path)

# Load pre-determined SMILES strings from a CSV file
def load_predetermined_smiles(csv_path):
    df_smiles = pd.read_csv(csv_path)
    return df_smiles['SMILES'].tolist()

predetermined_smiles = load_predetermined_smiles('unique_smiles.csv')

# Custom dataset class
class MolecularDataset(Dataset):
    def __init__(self, df, predetermined_smiles):
        self.df = df
        self.predetermined_smiles = predetermined_smiles
        self.valid_indices = [i for i, frag in enumerate(df['FRAG_SMILES']) if frag in predetermined_smiles]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        drug_smiles = self.df.iloc[original_idx]['DRUG SMILES']
        fragment_smiles = self.df.iloc[original_idx]['FRAG_SMILES']

        # Convert SMILES to molecular graphs
        drug_mol = Chem.MolFromSmiles(drug_smiles)
        fragment_mol = Chem.MolFromSmiles(fragment_smiles)

        # Convert molecules to RDKit graphs and then to PyTorch Geometric Data objects
        drug_graph = self.mol_to_graph(drug_mol)
        target_idx = self.predetermined_smiles.index(fragment_smiles)

        return drug_graph, target_idx

    def mol_to_graph(self, mol):
        adj = GetAdjacencyMatrix(mol)
        edges = np.array(np.nonzero(adj)).T
        x = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.float).unsqueeze(1)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        return data

# GNN Model with Classification Layer
class GNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Pooling to obtain graph-level features
        x = self.dropout(x)
        x = self.linear(x)
        return x

# Initialize dataset and dataloader
dataset = MolecularDataset(df, predetermined_smiles)
dataloader = GeometricDataLoader(dataset, batch_size=32, shuffle=True)

# Model parameters
model = GNN(num_node_features=1, hidden_channels=128, num_classes=len(predetermined_smiles)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
loss_fn = nn.CrossEntropyLoss()  # CrossEntropy for classification

# Define number of epochs
num_epochs = 100  # Increased for better learning

# Training Loop
model.train()
train_losses = []

for epoch in range(num_epochs):
    epoch_losses = []  # Initialize list for each epoch
    for batch in dataloader:
        drug_graphs, target_indices = batch  # Unpack the batch (tuple)
        
        # Move each element to the device individually
        drug_graphs = drug_graphs.to(device)
        target_indices = target_indices.to(device)

        # Forward pass
        pred = model(drug_graphs.x, drug_graphs.edge_index, drug_graphs.batch)

        # Compute loss
        loss = loss_fn(pred, target_indices)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    scheduler.step()
    train_losses.append(np.mean(epoch_losses))

# Save the trained model
torch.save(model.state_dict(), "gnn_model.pth")

# Evaluation
model.eval()
correct_predictions = 0

with torch.no_grad():
    for batch in dataloader:
        drug_graphs, target_indices = batch  # Unpack the batch (tuple)
        
        # Move each part of the drug_graph to the device
        drug_graphs = drug_graphs.to(device)
        target_indices = target_indices.to(device)

        # Forward pass
        pred = model(drug_graphs.x, drug_graphs.edge_index, drug_graphs.batch)

        predicted_indices = pred.argmax(dim=1)
        correct_predictions += (predicted_indices == target_indices).sum().item()

accuracy = correct_predictions / len(dataset)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Plot the training loss
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()

# Print final evaluation metrics
print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
