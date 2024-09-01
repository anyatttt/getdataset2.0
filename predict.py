import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd

# Define the GNN model architecture
class GNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, output_size):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, output_size)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

# Define the MolecularDataset class
class MolecularDataset:
    def __init__(self, predetermined_smiles):
        self.predetermined_smiles = predetermined_smiles

    def mol_to_graph(self, mol):
        adj = Chem.GetAdjacencyMatrix(mol)
        edges = np.array(np.nonzero(adj)).T
        x = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.float).unsqueeze(1)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        return data

    def graph_to_smiles(self, node_features, edge_index):
        mol = Chem.RWMol()
        atom_indices = []

        # Add atoms to the molecule
        for atom_feature in node_features:
            atom_idx = mol.AddAtom(Chem.Atom(int(atom_feature.item())))
            atom_indices.append(atom_idx)

        # Track added bonds to avoid duplicates
        added_bonds = set()

        # Add bonds to the molecule
        for i in range(edge_index.size(1)):
            start_idx, end_idx = int(edge_index[0, i].item()), int(edge_index[1, i].item())
            if start_idx < len(atom_indices) and end_idx < len(atom_indices):
                bond = (min(start_idx, end_idx), max(start_idx, end_idx))
                if bond not in added_bonds:
                    try:
                        mol.AddBond(atom_indices[start_idx], atom_indices[end_idx], Chem.BondType.SINGLE)
                        added_bonds.add(bond)
                    except Exception as e:
                        print(f"Warning: Failed to add bond between {start_idx} and {end_idx}: {e}")
                else:
                    print(f"Warning: Duplicate bond between {start_idx} and {end_idx} skipped.")
            else:
                print(f"Warning: Invalid bond between {start_idx} and {end_idx}")

        try:
            Chem.SanitizeMol(mol)  # Sanitize the molecule to ensure it is valid
            return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f"Warning: Sanitization failed: {e}")
            return None

    def find_closest_smiles(self, predicted_smiles):
        # Compare predicted SMILES with predetermined ones and return the closest match
        closest_smiles = None
        min_distance = float('inf')
        for smile in self.predetermined_smiles:
            dist = Chem.MolToInchi(Chem.MolFromSmiles(predicted_smiles)) == Chem.MolToInchi(Chem.MolFromSmiles(smile))
            if dist < min_distance:
                min_distance = dist
                closest_smiles = smile
        return closest_smiles

# Load the trained model
output_size = 9044  # This should match the output size used during training
model = GNN(num_node_features=1, hidden_channels=128, output_size=output_size)
model.load_state_dict(torch.load("gnn_model.pth", map_location=torch.device('cpu')))
model.eval()

# Load the predetermined SMILES
def load_predetermined_smiles(file_path):
    """
    Load predetermined SMILES strings from a CSV file.
    Assumes the CSV file has a column 'SMILES'.
    """
    df = pd.read_csv(file_path)
    return df['SMILES'].tolist()

predetermined_smiles = load_predetermined_smiles('unique_smiles.csv')  # Adjust the path as needed
dataset = MolecularDataset(predetermined_smiles)

# Define a function to predict the fragment from a given drug SMILES
def predict_fragment(drug_smiles):
    mol = Chem.MolFromSmiles(drug_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {drug_smiles}")

    drug_graph = dataset.mol_to_graph(mol)
    drug_graph = drug_graph.to(torch.device('cpu'))

    with torch.no_grad():
        pred = model(drug_graph.x, drug_graph.edge_index, torch.tensor([0] * drug_graph.num_nodes))  # Dummy batch
        pred_atom_indices = torch.argmax(pred, dim=1).tolist()  # Get the most likely atom indices
        pred_smiles = dataset.graph_to_smiles(torch.tensor(pred_atom_indices, dtype=torch.float32).unsqueeze(1), drug_graph.edge_index)
        return dataset.find_closest_smiles(pred_smiles)

# Example usage
if __name__ == "__main__":
    drug_smiles = input("Enter the drug SMILES: ")
    try:
        fragment_smiles = predict_fragment(drug_smiles)
        if fragment_smiles:
            print(f"Predicted Fragment SMILES: {fragment_smiles}")
        else:
            print("Failed to generate a valid SMILES string for the fragment.")
    except Exception as e:
        print(f"Error: {e}")
