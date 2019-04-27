import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import EdgeSoftmax
from MLPGAT import MLPGAT
from MLPGAT_1 import MLPGAT_1
from MLPGAT_average import MLPGAT_average

class MLP(nn.Module):
    def __init__(self, num_classes, alpha):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * num_classes, num_classes) 
        self.leaky_relu = nn.LeakyReLU(alpha)
        nn.init.xavier_normal_(self.fc1.weight.data, gain=1.414)

    def forward(self, x):
        out = self.fc1(x)
        out = self.leaky_relu(out)
        return out

class ENSEMBLE(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual):
        super(ENSEMBLE, self).__init__()
        self.g = g
        # input projection (no residual)

        self.MLPGAT = MLPGAT(g, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, alpha, residual)
        self.MLPGAT_1 = MLPGAT_1(g, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, alpha, residual)
        self.MLPGAT_average = MLPGAT_average(g, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, alpha, residual)
        self.mlp = MLP(num_classes, alpha)

    def forward(self, inputs):
        h = inputs
        res1 = self.MLPGAT(h)
        res2 = self.MLPGAT_1(h)
        res3 = self.MLPGAT_average(h)
        res = torch.cat([res1, res2], 1)
        res = torch.cat([res, res3], 1)
        # output projection
        logits = self.mlp(res)
        return logits