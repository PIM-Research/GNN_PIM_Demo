from subprocess import call

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models import SAGE, NamedGCNConv, wage_init_, Q, WAGERounding, C
from util import train_decorator
from util.global_variable import args, run_recorder, weight_quantification, grad_clip, grad_quantiication
from util.hook import set_vertex_map, set_updated_vertex_map
from util.logger import Logger
from util.other import transform_adj_matrix, transform_matrix_2_binary, store_updated_list_and_adj_matrix, norm_adj
from util.train_decorator import TrainDecorator


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
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            count += 1
            self.convs.append(
                NamedGCNConv(hidden_channels, hidden_channels, cached=True, name='convs.' + str(count) + '.gcn_conv',
                             adj_activity=adj_activity))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        count += 1
        self.convs.append(
            NamedGCNConv(hidden_channels, out_channels, cached=True, name='convs.' + str(count) + '.gcn_conv',
                         adj_activity=adj_activity))
        # 初始化神经网络权重，并对其进行量化
        self.weight_scale = {}
        self.weight_acc = {}
        if self.bits_W != -1:
            for name, param in self.named_parameters():
                if 'weight' in name and 'convs' in name:
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
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
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
        return x.log_softmax(dim=-1)

    def record_cost(self):
        message_cost = round(self.convs[0].message_cost_record, 2)
        aggregate_cost = round(self.convs[0].aggregate_cost_record, 2)
        message_proportion = round(message_cost / (message_cost + aggregate_cost), 2)
        aggregate_proportion = round(aggregate_cost / (message_cost + aggregate_cost), 2)

        with open('record/gcn.txt', 'a') as gcn:
            gcn.write(
                f'gcn record: message_cost {message_cost} {message_proportion} aggregate_cost {aggregate_cost} {aggregate_proportion}\n')


def train(model, data, train_idx, optimizer, train_decorator: TrainDecorator, cur_epoch=0, cluster_label=None):
    if args.call_neurosim:
        train_decorator.create_bash_command(cur_epoch, model.bits_W, model.bits_A)
    model.train()
    # 量化权重
    if args.bl_weight != -1:
        train_decorator.quantify_weight(model, 1, cur_epoch)

    # 绑定钩子函数，记录各层的输入
    if args.call_neurosim:
        train_decorator.bind_hooks(model, 1, cur_epoch)

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[cluster_label[train_idx]] if cluster_label is not None else \
        model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()

    # 量化梯度
    if args.bl_grad != -1:
        train_decorator.quantify_grad(model)

    optimizer.step()
    # 清除钩子
    if args.call_neurosim:
        train_decorator.clear_hooks(model, 1, cur_epoch)

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, cluster_label=None):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_idx = cluster_label[split_idx['train']] if cluster_label is not None else split_idx['train']
    valid_idx = cluster_label[split_idx['valid']] if cluster_label is not None else split_idx['valid']
    test_idx = cluster_label[split_idx['test']] if cluster_label is not None else split_idx['test']

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[train_idx],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[valid_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[test_idx],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    train_dec = train_decorator.TrainDecorator(weight_quantification, grad_quantiication, grad_clip, run_recorder)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    adj_matrix = norm_adj(data.adj_t) if args.call_neurosim else None
    # 获取词嵌入数量
    cluster_label = None
    if args.use_cluster:
        cluster_label, data.num_nodes, adj_matrix, data.adj_t = transform_adj_matrix(data, device)

    if args.call_neurosim:
        adj_coo = adj_matrix.coo()
        adj_stack = torch.stack([adj_coo[0], adj_coo[1], adj_coo[2]])
    else:
        adj_stack = None

    # 获取顶点特征更新列表
    updated_vertex, vertex_pointer = store_updated_list_and_adj_matrix(adj_t=data.adj_t, adj_binary=adj_stack)
    if vertex_pointer is not None:
        set_vertex_map(vertex_pointer)
    set_updated_vertex_map(updated_vertex)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout,
                    bl_weight=args.bl_weight, bl_activate=args.bl_activate, bl_error=args.bl_error,
                    recorder=run_recorder).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    if args.percentile != 0:
        train_dec.bind_update_hook(model)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer, train_decorator=train_dec, cur_epoch=epoch,
                         cluster_label=cluster_label)
            result = test(model, data, split_idx, evaluator, cluster_label=cluster_label)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
            if args.call_neurosim:
                call(["chmod", "o+x", run_recorder.bootstrap_path])
                call(["/bin/bash", run_recorder.bootstrap_path])
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
