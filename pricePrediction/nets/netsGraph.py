from collections import OrderedDict
from typing import Iterable

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pricePrediction import config
from pricePrediction.ArgParser_base import ArgParseable
from pricePrediction.nets.FDS_imbalance import FDS


class PricePredictorModule(pl.LightningModule, ArgParseable):

    DESIRED_PARAMS_TO_ASK= ['n_layers', 'hidden_size_node', 'hidden_size_edges', 'fcc_hidden_size', 'use_fds', 'lr',
                            'dropout', 'gnn_class', 'weight_decay', 'training_loss']
    def __init__(
        self,
        n_layers: int = config.N_LAYERS,
        hidden_size_node: int = config.N_HIDDEN_NODE,
        hidden_size_edges: int = config.N_HIDDEN_EDGE,
        gnn_class="GNN_PNAConv",
        towers: int = 5,
        fcc_hidden_size: int = 25,
        dropout: float = 0,
        use_fds: bool = False,
        training_loss : str= "l2", # l1, l2 or huber
        last_layer_size: int = 1,

        nodes_n_features: int = None,
        edges_n_features: int = None,
        deg: Iterable[float] = None,

        lr: float = config.LEARNING_RATE,
        b1: float = 0.5,
        b2: float = 0.999,
        weight_decay=1e-8,
        data_hparams =None,
    ):


        '''

        :param str encodedDir: The directory where the dataset has been prepared as lmdb files. defaults to %(config.ENCODED_DIR)s
        :param str deg_fname: The file containing the statistics about the nodes degree. By default is automatically searched within
                              encodedDir defaults to None
        :param int n_layers: The number of layers.
        :param int hidden_size_node: Channels for nodes.
        :param int hidden_size_edges: Channels for edges.
        :param str gnn_class: Name of the model type.
        :param int towers: Only for gnn_class="GNN_PNAConv", The number of towers of the model.
        :param int fcc_hidden_size: Size of fully connected.
        :param float dropout: Dropout rate.
        :param bool use_fds: Use feature density smoothing for data imbalance
        :param str training_loss: Training loss: "l2" "l1" or "huber"
        :param int last_layer_size: The number of neurons for last layer. Set it to 1 for regression
        :param int nodes_n_features: The number of features used to encode a node
        :param int edges_n_features: The number of features used to encode an edge
        :param Iterable deg: The degree count of the dataset
        :param float lr: Learning rate
        :param float b1: Adam optimizer b1
        :param float b2: Adam optimizer b1
        '''

        super().__init__()
        self.save_hyperparameters() #copies __init__() kwargs to self.hparams
        deg =  self.hparams.deg
        if isinstance(deg, list):
            deg = torch.tensor(deg[:], dtype=torch.long)
        else:
            deg = deg.clone().detach()
        self.hparams.deg = deg

        if gnn_class == "GNN_PNAConv":
            from pricePrediction.nets.basicArchitectures import GNN_PNAConv
            GNN = GNN_PNAConv
        elif gnn_class == "GNN_AttentiveFP":
            from pricePrediction.nets.basicArchitectures import GNN_AttentiveFP
            GNN = GNN_AttentiveFP
        else:
            raise ValueError("Error, gnn_class not supported")

        self.net = GNN(** self.hparams)

        self.dropoutLayer = nn.Dropout(self.hparams.dropout)

        self.final_layer = nn.Linear(self.net.fcc_hidden_size, 1)

        if use_fds:
            self.fds = FDS(self.net.latent_size, max_val=10, start_update=1)
        else:
            self.fds = None
        if training_loss=="l1":
            self.trainLossF = F.l1_loss
        elif training_loss=="l2":
            self.trainLossF = F.mse_loss
        elif training_loss=="huber":
            self.trainLossF = F.smooth_l1_loss
        else:
            raise ValueError("Loss option not recognized")

    def compute_training_loss(self, y_pred, y, w=None):
        loss = self.trainLossF(y_pred, y, reduction='none')
        if w is not None:
            loss *= w
        loss = torch.mean(loss)
        return loss

    def forward(self, g):
        return self.compute_y_pred( g, training=False)

    def compute_y_pred(self, g, y=None, training=True):
        x = self.net(g.x, g.edge_index, g.edge_attr, g.batch)
        if training and self.fds is not None:
            self.fds.smooth(x, y, self.current_epoch)
        x = self.dropoutLayer(x)
        y_pred = self.final_layer(x)
        return y_pred.view(-1)

    def resolve_batch(self, batch):
        if isinstance(batch, list):
            return batch[0]
        else:
            return batch

    def training_step(self, batch, batch_idx):
        graphs = self.resolve_batch(batch)
        y_pred = self.compute_y_pred(graphs, training=True)
        loss = self.compute_training_loss(y_pred, graphs.y, graphs.w)

        loss_l1 = F.l1_loss(y_pred, graphs.y)

        self.log('loss', loss)
        self.log('loss_l1', loss_l1, prog_bar=True)

        tqdm_dict = {'loss': loss.detach()}
        output = OrderedDict({
            'loss': loss,
            'targets': graphs.y,
            'preds': y_pred.detach(),
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def _validation_step(self, graphs, batch_idx):
        with torch.no_grad():
            y_pred = self.compute_y_pred(graphs, training=False)
            loss = self.compute_training_loss(y_pred, graphs.y, graphs.w)
            loss_l1 = F.l1_loss(y_pred, graphs.y)
            return y_pred, loss, loss_l1

    def validation_step(self, batch, batch_idx):
        graphs = self.resolve_batch(batch)
        y_pred, loss, loss_l1 = self._validation_step(graphs, batch_idx)
        # tensorboard = self.logger.experiment
        # tensorboard.add_histogram("val_price", graphs.y)
        # tensorboard.add_histogram("val_preds", y_pred)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True) #rank_zero_only=True
        self.log('val_lossL1', loss_l1, prog_bar=True, sync_dist=True)
        return loss

    def training_epoch_end(self, outputs ):
        # tensorboard = self.logger.experiment
        # tensorboard.add_histogram("train_price", y)
        # tensorboard.add_histogram("train_preds", torch.cat([ x["preds"] for x in  outputs]))
        # tensorboard.add_histogram("train_price", torch.cat([ x["targets"] for x in  outputs]))
        print("Epoch %d done"%self.current_epoch)

    def test_step(self, batch, batch_idx):
        graphs = self.resolve_batch(batch)
        y_pred, loss, loss_l1 = self._validation_step(graphs, batch_idx)
        tensorboard = self.logger.experiment
        # tensorboard.add_histogram("test_price", graphs.y)
        # tensorboard.add_histogram("test_preds", y_pred)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.log('test_lossL1', loss_l1, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(b1, b2), weight_decay=self.hparams.weight_decay)
        # for param_group in opt.param_groups:
        #     print(param_group["lr"])

        conf = {
            'optimizer': opt,
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, cooldown=2,
                                                                patience=config.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS, ),
            'monitor': 'val_loss'
        }

        return conf

if __name__ == "__main__":
    from pricePrediction.preprocessData.smilesToGraph import smiles_to_graph
    from torch_geometric.data import Batch

    degree =[0, 41130, 117278, 70152, 3104]

    g = smiles_to_graph("CCCCCCCCCCC")
    nodes_n_features = g.x.shape[1]
    edges_n_features = g.edge_attr.shape[1]
    gnn_class = "GNN_PNAConv"# "GNN_AttentiveFP"
    net = PricePredictorModule(nodes_n_features=nodes_n_features, edges_n_features=edges_n_features,
                               deg=degree, lr=1e-3, gnn_class=gnn_class)

    print( net )

    graphs = [smiles_to_graph(smi) for smi in ["CCCCCO", "CCCCCNCCCCN"]]
    print(graphs[0])
    print( net(Batch.from_data_list(graphs)))
