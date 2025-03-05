# %%
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from HIL import HIL


class LamNet(nn.Module):
    def __init__(self, node_dim1, node_dim2, hidden_dim):
        """
        LamNet model for predicting binding free energies
        
        Args:
            node_dim1: Input dimension of node features
            node_dim2: Input dimension for auxiliary description
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial node feature transformation
        self.lin_node1 = nn.Sequential(Linear(node_dim1, hidden_dim), nn.SiLU())
    
        # Graph convolution layers
        self.gconv1 = HIL(hidden_dim, hidden_dim)
        self.gconv2 = HIL(hidden_dim, hidden_dim)
        self.gconv3 = HIL(hidden_dim, hidden_dim)

        self.bn = nn.BatchNorm1d(1)
        
        # Feature fusion networks
        self.fnn = FNN(hidden_dim, hidden_dim)
        self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)

        # Embedding for auxiliary description
        self.up = nn.Embedding(node_dim2, 256)

    def forward(self, data1, data2, extra):
        """
        Forward pass
        
        Args:
            data1: First molecule graph data
            data2: Second molecule graph data 
            extra: Auxiliary description tensor
            
        Returns:
            Predicted binding free energy
        """
        # Extract graph features
        x1, edge_index_intra1, edge_index_inter1, pos1 = \
        data1.x, data1.edge_index_intra, data1.edge_index_inter, data1.pos
        x2, edge_index_intra2, edge_index_inter2, pos2 = \
        data2.x, data2.edge_index_intra, data2.edge_index_inter, data2.pos
            
        # ligand1_with/without_pocket
        x1 = self.lin_node1(x1)
        # encoder
        x1 = self.gconv1(x1, edge_index_intra1, edge_index_inter1, pos1)
        x1 = self.gconv2(x1, edge_index_intra1, edge_index_inter1, pos1)
        x1 = self.gconv3(x1, edge_index_intra1, edge_index_inter1, pos1)
        # Global pooling
        x1 = global_add_pool(x1, data1.batch)

        # ligand2_with_pocket
        x2 = self.lin_node1(x2)
        # encoder
        x2 = self.gconv1(x2, edge_index_intra2, edge_index_inter2, pos2)
        x2 = self.gconv2(x2, edge_index_intra2, edge_index_inter2, pos2)
        x2 = self.gconv3(x2, edge_index_intra2, edge_index_inter2, pos2)
        # Global pooling
        x2 = global_add_pool(x2, data2.batch)

        # Process auxiliary description
        extra1 = 1 - torch.matmul(extra, self.up.weight)
        extra2 = torch.matmul(extra, self.up.weight)

        # Feature multiplication with auxiliary info
        x1 = torch.mul(x1, extra1)
        x2 = torch.mul(x2, extra2)

        # Concatenate and process graph features
        y = self.fnn(
            torch.cat([x1.to(torch.float32), x2.to(torch.float32), (x1-x2).to(torch.float32)], dim=-1))

        # Final prediction
        y = self.fc(y)

        return y.view(-1)


class FNN(nn.Module):
    """
    Feature fusion network to process concatenated graph features
    
    Args:
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
    """
    def __init__(self, hidden_dim, output_dim):
        super(FNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Multi-layer feed forward network
        ffn = [nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.ReLU()]
        ffn.append(nn.Linear(hidden_dim * 2, hidden_dim))
        ffn.append(nn.ReLU())
        ffn.append(nn.Linear(hidden_dim, int(hidden_dim * 0.5)))
        ffn.append(nn.ReLU())
        ffn.append(nn.Linear(int(hidden_dim * 0.5), output_dim))

        self.FNN = nn.Sequential(*ffn)
    
    def forward(self, h):
        for layer in self.FNN:
            h = layer(h)
        return h


class FC(nn.Module):
    """
    Final fully connected layers for prediction
    
    Args:
        d_graph_layer: Input dimension from graph features
        d_FC_layer: Hidden layer dimension
        n_FC_layer: Number of FC layers
        dropout: Dropout rate
        n_tasks: Number of prediction tasks
    """
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        
        # Build FC layers with dropout, activation and batch norm
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h

# %%