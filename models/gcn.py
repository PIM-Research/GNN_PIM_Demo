import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .wage_quantizer import WAGERounding, Q, C
from .wage_initializer import wage_init_
from .named_gcnconv import NamedGCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, bl_weight=-1, bl_activate=-1, bl_error=-1, writer=None, recorder=None, adj_activity=0):
        super(GCN, self).__init__()
        print(bl_weight, bl_activate, bl_error)
        self.bits_W = bl_weight
        self.bits_A = bl_activate
        self.bits_E = bl_error
        self.writer = writer
        self.adj_activity = adj_activity

        count = 0
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            NamedGCNConv(in_channels, hidden_channels, cached=True, name='convs.' + str(count) + '.gcn_conv',
                         adj_activity=adj_activity))
        for _ in range(num_layers - 2):
            count += 1
            self.convs.append(
                NamedGCNConv(hidden_channels, hidden_channels, cached=True, name='convs.' + str(count) + '.gcn_conv',
                             adj_activity=adj_activity))
        count += 1
        self.convs.append(
            NamedGCNConv(hidden_channels, out_channels, cached=True, name='convs.' + str(count) + '.gcn_conv',
                         adj_activity=adj_activity))
        # 初始化神经网络权重，并对其进行量化
        self.weight_scale = {}
        self.weight_acc = {}
        if self.bits_W != -1:
            for name, param in self.named_parameters():
                if 'weight' in name:
                    data_before = param.data.T
                    wage_init_(param, bl_weight, name, self.weight_scale, factor=1.0)
                    self.weight_acc[name] = Q(param.data, bl_weight)
                    if recorder is not None:
                        recorder.record_change('layer_init', name, data_before, self.weight_acc[name].T,
                                               delimiter=',', fmt='%10f')

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.gcn_conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            # 对激活值输出进行量化
            if self.bits_A != -1:
                x = C(x, self.bits_A)  # keeps the gradients
            x = WAGERounding.apply(x, self.bits_A, self.bits_E, None)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        # 对激活值输出进行量化
        if self.bits_A != -1:
            x = C(x, self.bits_A)  # keeps the gradients
        x = WAGERounding.apply(x, self.bits_A, self.bits_E, None)
        return x

    def record_cost(self):
        message_cost = round(self.convs[0].message_cost_record, 2)
        aggregate_cost = round(self.convs[0].aggregate_cost_record, 2)
        message_proportion = round(message_cost / (message_cost + aggregate_cost), 2)
        aggregate_proportion = round(aggregate_cost / (message_cost + aggregate_cost), 2)

        with open('record/gcn.txt', 'a') as gcn:
            gcn.write(
                f'gcn record: message_cost {message_cost} {message_proportion} aggregate_cost {aggregate_cost} {aggregate_proportion}\n')
