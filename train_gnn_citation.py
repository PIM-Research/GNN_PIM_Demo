import argparse
from subprocess import call

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from util.logger import Logger

from models.gcn import GCN
from tensorboardX import SummaryWriter
from util.hook import set_vertex_map, set_updated_vertex_map
from util.global_variable import args, run_recorder, weight_quantification, grad_clip, grad_quantiication
from util.other import get_updated_num, norm_adj, quantify_adj, store_updated_list_and_adj_matrix, record_net_structure, \
    record_pipeline_prediction_info
from util.train_decorator import TrainDecorator
import time


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


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, data, split_edge, optimizer, batch_size, train_decorator: TrainDecorator, cur_epoch=0):
    if args.call_neurosim:
        train_decorator.create_bash_command(cur_epoch, model.bits_W, model.bits_A)
    model.train()
    predictor.train()

    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)

    total_loss = total_examples = 0
    dst_vertex_num = 0
    num_i = 0
    for i, perm in enumerate(DataLoader(range(source_edge.size(0)), batch_size,
                                        shuffle=True)):
        start_time = time.perf_counter()
        # 量化权重
        if args.bl_weight != -1:
            train_decorator.quantify_weight(model, i, cur_epoch)

        # 绑定钩子函数，记录各层的输入
        if args.call_neurosim:
            train_decorator.bind_hooks(model, i, cur_epoch)

        src, dst = source_edge[perm], target_edge[perm]
        dst_vertex_num += get_updated_num(torch.unique(dst))
        num_i += 1
        if args.filter_adj:
            data.adj_t = train_decorator.filter_adj_by_batch(adj_t=data.adj_t, source_vertexes=src,
                                                             dst_vertexes=dst, batch_index=i)

        optimizer.zero_grad()
        h = model(data.x, data.adj_t)
        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.num_nodes, src.size(),
                                dtype=torch.long, device=h.device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        # 量化梯度
        if args.bl_weight != -1:
            train_decorator.quantify_grad(model)
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        # 清除钩子
        if args.call_neurosim:
            train_decorator.clear_hooks(model, i, cur_epoch)
        end_time = time.perf_counter()
        print(f'current epoch:{cur_epoch}  current Iteration:{i} epoch time:{end_time - start_time}')
    print('dst_vertex_num_avg:', dst_vertex_num / num_i * (1 - 0.01 * args.percentile))
    print('num_i:', num_i)

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    predictor.eval()

    h = model(data.x, data.adj_t)

    def test_split(split):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr


def main():
    writer = SummaryWriter()
    train_dec = TrainDecorator(weight_quantification, grad_quantiication, grad_clip, run_recorder)
    # runs为10，epochs为50，神经网络层数为3层，dropout为0，lr为0.0005，隐藏层数为256，batch_size为64*1024
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-citation2',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
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
        record_net_structure(data.num_nodes, data.num_features, args.hidden_channels, args.hidden_channels,
                             args.num_layers)
    data = data.to(device)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout, bl_weight=args.bl_weight, bl_activate=args.bl_activate, bl_error=args.bl_error,
                    recorder=run_recorder,normalize=False).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-citation2')
    logger = Logger(args.runs, args)
    if args.percentile != 0:
        train_dec.bind_update_hook(model)
    test_time = 0
    epoch_time = 0

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            start_time = time.perf_counter()
            loss = train(model, predictor, data, split_edge, optimizer, args.batch_sizet, train_decorator=train_dec,
                         cur_epoch=epoch)
            writer.add_scalar('citation/Loss', loss, epoch)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            if epoch % args.eval_steps == 0:
                test_s = time.perf_counter()
                result = test(model, predictor, data, split_edge, evaluator,
                              args.batch_size)
                test_e = time.perf_counter()
                test_time += test_e - test_s
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_mrr, valid_mrr, test_mrr = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {train_mrr:.4f}, '
                          f'Valid: {valid_mrr:.4f}, '
                          f'Test: {test_mrr:.4f}')
                    writer.add_scalar('Train accuracy', train_mrr, epoch)
                    writer.add_scalar('Valid accuracy', valid_mrr, epoch)
                    writer.add_scalar(' Test accuracy', test_mrr, epoch)
                    print('---')
            record_pipeline_prediction_info(data.num_nodes, data.num_nodes,  data.num_features, args.hidden_channels,
                                            args.hidden_channels, epoch, 3)
            if args.call_neurosim:
                call(["chmod", "o+x", run_recorder.bootstrap_path])
                call(["/bin/bash", run_recorder.bootstrap_path])
            end_time = time.perf_counter()
            epoch_time += start_time - end_time
            print('epoch运行时长：', epoch_time - test_time, '秒')
            print('num_nodes:', data.num_nodes, 'input_channels:', data.num_features, 'hidden_channels:',
                  args.hidden_channels, 'output_channels:', args.hidden_channels)

        print('GraphSAGE' if args.use_sage else 'GCN')
        logger.print_statistics(run)
    print('GraphSAGE' if args.use_sage else 'GCN')
    logger.print_statistics()
    print('运行时长：', epoch_time - test_time, '秒')


if __name__ == "__main__":
    main()
