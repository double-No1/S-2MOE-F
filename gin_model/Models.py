import torch.nn as nn
import torch
from gin_model.GraphNets import GIN,GraphNorm,GIN_classifier,GINConv
import torch.nn.init as init
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool,global_max_pool


class OCGIN(nn.Module):
    def __init__(self, dim_features, config):
        super(OCGIN, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.dim_targets = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.net = GIN(dim_features, self.dim_targets, config)
        self.center = torch.zeros(1, self.dim_targets * self.num_layers, requires_grad=False).to('cuda')
        self.reset_parameters()
    def forward(self, data):
        data = data.to(self.device)
        # print(data)
        # print(data.edge_index)
        z = self.net(data)
        # print("z", z)
        return z, self.center

    def init_center(self, train_loader):
        with torch.no_grad():
            for data in train_loader:
                data = data.to('cuda')
                z = self.forward(data)
                self.center += torch.sum(z[0], 0, keepdim=True)
            self.center = self.center / len(train_loader.dataset)

    def reset_parameters(self):
        self.net.reset_parameters()

