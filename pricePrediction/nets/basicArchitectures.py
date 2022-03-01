import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool, AttentiveFP, Set2Set
from torch.nn import ModuleList, Linear, ReLU, Sequential, Dropout


class MLP(nn.Module):

    def __init__(self,dims, n_layers, hidden_size, dropout=0 ):

        super().__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dims = dims
        self.dropout = dropout


        def block(in_size, n_hidden):
            layers = [
                nn.Linear(in_size, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
            ]
            if self.dropout > 0:
                layers.append(
                    nn.Dropout(self.dropout),
                )
            return layers

        # Define PyTorch model
        self.model = nn.Sequential(
            *block(np.prod(dims), self.hidden_size)
        )

        self.latent_size = self.hidden_size

    def forward(self, x):
        return self.model(x)


class GNN_PNAConv(torch.nn.Module):
    def __init__(self,
        nodes_n_features: int,
        edges_n_features: int,
        deg: torch.tensor ,
        n_layers: int = 6,
        hidden_size_node: int = 75,
        hidden_size_edges: int = 50,
        towers: int = 5,
        fcc_hidden_size: int = 50,
        dropout: float = 0,
        use_fds: bool = False,
                 **args
        ):
        super(GNN_PNAConv, self).__init__()

        self.node_emb = MLP(nodes_n_features, 1,  hidden_size_node)
        self.edge_emb = MLP(edges_n_features, 1, hidden_size_edges)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(n_layers):
            conv = PNAConv(in_channels=hidden_size_node, out_channels=hidden_size_node,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=hidden_size_edges, towers=towers, pre_layers=2, post_layers=2,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size_node))
        self.set2set = Set2Set(hidden_size_node, processing_steps=6)
        fc_layers = [
            Linear(2*hidden_size_node, hidden_size_node), BatchNorm(hidden_size_node), ReLU(),
            Linear(hidden_size_node, fcc_hidden_size), BatchNorm(fcc_hidden_size), ReLU(),
        ]
        self.fcc_hidden_size = fcc_hidden_size
        if dropout>0:
            fc_layers += [ Dropout(p=dropout) ]

        self.mlp = Sequential( *fc_layers )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = self.set2set(x, batch) #Set2Set #GlobalAttention #GraphMultisetTransformer

        return self.mlp(x)

class GNN_AttentiveFP(torch.nn.Module):
    def __init__(self,
        nodes_n_features: int,
        edges_n_features: int,
        deg: torch.tensor ,
        n_layers: int = 4,
        hidden_size_node: int = 75,
        hidden_size_edges: int = 50,
        towers: int = 5,
        fcc_hidden_size: int = 50,
        dropout: float = 0,
        use_fds: bool = False,
                 **args
        ):
        super(GNN_AttentiveFP, self).__init__()

        self.attent_net = AttentiveFP(nodes_n_features, hidden_size_node, hidden_size_node, edges_n_features,
                                      n_layers, n_layers, dropout= dropout)

        fc_layers = [
            Linear(hidden_size_node, hidden_size_edges), ReLU(),
            Linear(hidden_size_edges, fcc_hidden_size), ReLU(),
        ]
        self.fcc_hidden_size = fcc_hidden_size
        if dropout>0:
            fc_layers += Dropout(p=dropout)

        self.mlp = Sequential( *fc_layers )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.attent_net(x, edge_index, edge_attr, batch)
        return self.mlp(x)
