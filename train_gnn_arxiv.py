from subprocess import call

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models import SAGE, GCN
from util import train_decorator
from util.global_variable import args, run_recorder, weight_quantification, grad_clip, grad_quantiication
from util.logger import Logger
from util.other import transform_adj_matrix, transform_matrix_2_binary, store_updated_list_and_adj_matrix, norm_adj
from util.train_decorator import TrainDecorator


def train(model, data, train_idx, optimizer, train_decorator: TrainDecorator, cur_epoch=0, cluster_label=None):
    if args.call_neurosim:
        train_decorator.create_bash_command(cur_epoch, model.bits_W, model.bits_A)
    model.train()
    # 量化权重
    train_decorator.quantify_weight(model, 1, cur_epoch)

    # 绑定钩子函数，记录各层的输入
    if args.call_neurosim:
        train_decorator.bind_hooks(model, 1, cur_epoch)

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[cluster_label[train_idx]]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()

    # 量化梯度
    train_decorator.quantify_activation(model)

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

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[cluster_label[split_idx['train']]],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[cluster_label[split_idx['valid']]],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[cluster_label[split_idx['test']]],
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

    adj_matrix = norm_adj(data.adj_t).to_dense().numpy()
    # 获取词嵌入数量
    cluster_label = None
    if args.use_cluster:
        cluster_label, data.num_nodes, adj_matrix, data.adj_t = transform_adj_matrix(data, device)

    # 转换为2进制
    adj_binary, activity = transform_matrix_2_binary(adj_matrix)

    # 获取顶点特征更新列表
    store_updated_list_and_adj_matrix(adj_t=data.adj_t, adj_binary=adj_binary)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout, bl_weight=args.bl_weight, bl_activate=args.bl_activate, bl_error=args.bl_error,
                    recorder=run_recorder, adj_activity=activity).to(device)

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
