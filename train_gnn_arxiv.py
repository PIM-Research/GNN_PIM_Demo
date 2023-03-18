import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from util import train_decorator
from util.logger import Logger
from models import GAT, GCN, SAGE, QW, QG, C
from util.global_variable import *
import numpy as np


def train(model, data, train_idx, optimizer, train_dec: train_decorator.TrainDecorator, cur_epoch):
    train_dec.create_bash_command(cur_epoch, args.bl_weight, args.bl_activation)
    model.train()
    # 量化权重
    train_dec.quantify_weight(model, 0, cur_epoch)
    # 绑定钩子函数，记录各层的输入
    train_dec.bind_hooks(model, cur_epoch)
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    # 量化梯度
    train_dec.quantify_activation(model)
    optimizer.step()
    # 清除钩子
    train_dec.clear_hooks(model, 0, cur_epoch)
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
    print(args)
    # 定义量化权重和梯度的lambda函数以及权重clip函数
    weight_quantification = lambda x, scale: QW(x, args.bl_weight, scale)
    grad_quantiication = lambda x: QG(x, args.bl_grad, args.bl_rand, args.lr)
    grad_clip = lambda x: C(x, args.bl_weight)

    train_dec = train_decorator.TrainDecorator(weight_quantification, grad_quantiication, grad_clip, run_recorder)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    adj_matrix = data.adj_t.set_diag().to_dense().numpy()
    run_recorder.record('', 'adj_matrix.csv', adj_matrix, delimiter=',', fmt='%d')
    activity = np.sum(adj_matrix.astype(np.float), axis=None) / np.size(adj_matrix)
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    elif args.use_gcn:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout, bl_weight=args.bl_weight, bl_activate=args.bl_activate, bl_error=args.bl_error,
                    recorder=run_recorder, adj_activity=activity).to(device)
    else:
        model = GAT(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer, train_dec, epoch)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
        model.record_cost()

    logger.print_statistics()


if __name__ == "__main__":
    main()
