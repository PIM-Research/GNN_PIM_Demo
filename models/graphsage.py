import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

    def record_cost(self):
        message_cost = round(self.convs[0].message_cost_record, 2)
        aggregate_cost = round(self.convs[0].aggregate_cost_record, 2)
        message_proportion = round(message_cost / (message_cost + aggregate_cost), 2)
        aggregate_proportion = round(aggregate_cost / (message_cost + aggregate_cost), 2)

        with open('record/sage.txt', 'a') as sage:
            sage.write(
                f'sage record: message_cost {message_cost} {message_proportion} aggregate_cost {aggregate_cost} {aggregate_proportion}\n')

