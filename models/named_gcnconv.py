from torch_geometric.nn import GCNConv
import torch


class NamedGCNConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True, bias: bool = True, name=None, bl_input=8,
                 bl_activate=8, bl_error=8, bl_weight=8, inference=0, on_off_ratio=10, cell_bit=1, adj_activity=0,
                 **kwargs):
        super().__init__()
        self.name = name
        self.bl_input = bl_input
        self.bl_activate = bl_activate
        self.bl_error = bl_error
        self.bl_weight = bl_weight
        self.inference = inference
        self.on_off_ratio = on_off_ratio
        self.cell_bit = cell_bit
        self.epoch = 0
        self.gcn_conv = GCNConv(in_channels, out_channels, improved, cached,
                                add_self_loops, normalize, bias, **kwargs)
        self.adj_activity = adj_activity

    def forward(self, x, adj_t):
        x = self.gcn_conv(x, adj_t)
        return x

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch
