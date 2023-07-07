import argparse
import time
from subprocess import call

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from tensorboardX import SummaryWriter
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models import NamedGCNConv, wage_init_, Q, C, WAGERounding
from util.global_variable import args, weight_quantification, grad_quantiication, run_recorder, grad_clip
from util.hook import set_vertex_map, set_updated_vertex_map
from util.logger import Logger
from util.other import norm_adj, quantify_adj, store_updated_list_and_adj_matrix, record_net_structure, get_updated_num, \
    record_pipeline_prediction_info
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
            NamedGCNConv(in_channels, hidden_channels, normalize=False, name='convs.' + str(count) + '.gcn_conv',
                         adj_activity=adj_activity))
        for _ in range(num_layers - 2):
            count += 1
            self.convs.append(
                NamedGCNConv(hidden_channels, hidden_channels, normalize=False, name='convs.' + str(count) + '.gcn_conv'
                             , adj_activity=adj_activity))
        count += 1
        self.convs.append(
            NamedGCNConv(hidden_channels, out_channels, normalize=False, name='convs.' + str(count) + '.gcn_conv',
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

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
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
        return torch.log_softmax(x, dim=-1)

    def record_cost(self):
        message_cost = round(self.convs[0].message_cost_record, 2)
        aggregate_cost = round(self.convs[0].aggregate_cost_record, 2)
        message_proportion = round(message_cost / (message_cost + aggregate_cost), 2)
        aggregate_proportion = round(aggregate_cost / (message_cost + aggregate_cost), 2)

        with open('record/gcn.txt', 'a') as gcn:
            gcn.write(
                f'gcn record: message_cost {message_cost} {message_proportion} aggregate_cost {aggregate_cost} '
                f'{aggregate_proportion}\n')


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
        return torch.log_softmax(x, dim=-1)


def train(model, data, train_idx, optimizer, train_decorator: TrainDecorator, cur_epoch=0, ):
    if args.call_neurosim:
        train_decorator.create_bash_command(cur_epoch, model.bits_W, model.bits_A)
    model.train()
    # 量化权重
    if args.bl_weight != -1:
        train_decorator.quantify_weight(model, 0, cur_epoch)

    # 绑定钩子函数，记录各层的输入
    if args.call_neurosim:
        train_decorator.bind_hooks(model, 0, cur_epoch)

    if args.filter_adj:
        data.adj_t = train_decorator.filter_adj_by_batch(adj_t=data.adj_t, source_vertexes=train_idx,
                                                         dst_vertexes=train_idx, batch_index=0)

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()

    # 量化梯度
    if args.bl_grad != -1:
        train_decorator.quantify_grad(model)
    optimizer.step()

    # 清除钩子
    if args.call_neurosim:
        train_decorator.clear_hooks(model, 0, cur_epoch)

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    writer = SummaryWriter()
    train_dec = TrainDecorator(weight_quantification, grad_quantiication, grad_clip, run_recorder)
    # runs为10，epochs为300，神经网络层数为3层，dropout为0.5，lr为0.01，隐藏层数为256，batch_size为64*1024
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    # 将邻接矩阵正则化
    if args.call_neurosim:
        adj_matrix = norm_adj(data.adj_t)
        if args.bl_activate != -1:
            adj_matrix = quantify_adj(adj_matrix, args.bl_activate)
    else:
        adj_matrix = None

    # 获取顶点特征更新列表
    updated_vertex, vertex_pointer = store_updated_list_and_adj_matrix(adj_t=data.adj_t, adj_binary=adj_matrix)

    if vertex_pointer is not None:
        set_vertex_map(vertex_pointer)
    set_updated_vertex_map(updated_vertex)

    if args.call_neurosim:
        record_net_structure(data.num_nodes, data.num_features, args.hidden_channels, dataset.num_classes,
                             args.num_layers)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    print('参与Aggregation顶点数：', get_updated_num(train_idx) * (1 - 0.01 * args.percentile))

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout,
                    bl_weight=args.bl_weight, bl_activate=args.bl_activate, bl_error=args.bl_error,
                    recorder=run_recorder).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)
    if args.percentile != 0:
        train_dec.bind_update_hook(model)
    test_time = 0
    start_time = time.perf_counter()
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            writer.add_scalar('products/Loss', loss, epoch)
            test_s = time.perf_counter()
            result = test(model, data, split_idx, evaluator)
            test_e = time.perf_counter()
            test_time += test_e - test_s
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
                writer.add_scalar(f'products/Train accuracy', 100 * train_acc, epoch)
                writer.add_scalar(f'products/Valid accuracy', 100 * valid_acc, epoch)
                writer.add_scalar(f'products/Test accuracy', 100 * test_acc, epoch)
            record_pipeline_prediction_info(data.num_nodes, data.num_nodes, data.num_features, args.hidden_channels,
                                            dataset.num_classes, epoch, 3)
            if args.call_neurosim:
                call(["chmod", "o+x", run_recorder.bootstrap_path])
                call(["/bin/bash", run_recorder.bootstrap_path])

        logger.print_statistics(run)
    logger.print_statistics()
    end_time = time.perf_counter()
    print('运行时长：', end_time - start_time - test_time, '秒')


if __name__ == "__main__":
    main()
