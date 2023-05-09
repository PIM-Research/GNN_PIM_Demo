import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from models import GCN
from .other import filter_edges, store_adj_matrix
from .train_decorator import TrainDecorator
from .global_variable import args
from .definition import NEGS
import numpy as np


def train(model: GCN, predictor, x, adj_t, split_edge, optimizer, batch_size, train_decorator: TrainDecorator,
          cur_epoch=0, cluster_label=None, adj_origin=None):
    if NEGS(args.negs) is NEGS.CLUSTER:
        row, col, _ = adj_t.coo()
        vertex_num = x.size(0)
    else:
        row, col, _ = adj_origin.coo()
        source_max = adj_origin.size(dim=0)
        des_max = adj_origin.size(dim=1)
        vertex_num = max(source_max, des_max)
    edge_index = torch.stack([col, row], dim=0)
    if args.call_neurosim:
        train_decorator.create_bash_command(cur_epoch, model.bits_W, model.bits_A)
    model.train()
    predictor.train()
    # 获取训练集的边集，即图中存在的所有边的集合，类型是tensor，共两维，行数代表边数，列为源节点和目标节点
    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    # perm实际上是一个子图，类型是tensor，元素代表本次预测用到的边的编号，这里perm起到索引作用
    # 这个数据集边的数量为1067911，batch_size为65536，所以有17个batch
    dst_vertex_num = 0
    num_i = 0
    # np.savetxt(f'record/{cur_epoch}_x.csv', x.detach().cpu().numpy(), delimiter=',', fmt='%10f')
    for i, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size,
                                        shuffle=True)):
        # 量化权重
        if args.bl_weight != -1:
            train_decorator.quantify_weight(model, i, cur_epoch)

        # 绑定钩子函数，记录各层的输入
        if args.call_neurosim:
            train_decorator.bind_hooks(model, i, cur_epoch)

        # 根据perm这个索引获得本次预测需要的边
        edge = pos_train_edge[perm].t()
        dst_vertex_num += torch.unique(torch.cat([edge[0], edge[1]], dim=-1)).shape[0]
        num_i += 1

        if args.filter_adj:
            adj_t = train_decorator.filter_adj_by_batch(adj_t=adj_t, source_vertexes=edge[0], dst_vertexes=edge[1],
                                                        batch_index=i)

        optimizer.zero_grad()

        # 进行图卷积计算
        h = model(x, adj_t)

        # print('edge[0]:', edge[0], ' edge[1]:', edge[1])
        if args.use_cluster:
            src = cluster_label[edge[0]]
            dst = cluster_label[edge[1]]
        else:
            src = edge[0]
            dst = edge[1]
        # print('edge[0]:', edge[0], ' edge[1]:', edge[1])
        # 预测这两个顶点之间是否存在边，1代表存在，0为不存在
        pos_out = predictor(h[src], h[dst])
        # print('pos_out:', pos_out)
        # 计算损失函数的值
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        # print('pos_loss:', pos_loss)

        # 什么是负采样？
        edge = negative_sampling(edge_index, num_nodes=vertex_num,
                                 num_neg_samples=perm.size(0), method='dense')
        if args.use_cluster and (NEGS(args.negs) is not NEGS.CLUSTER):
            src = cluster_label[edge[0]]
            dst = cluster_label[edge[1]]
        else:
            src = edge[0]
            dst = edge[1]

        # 预测这两个顶点之间是否存在边，1代表存在，0为不存在
        neg_out = predictor(h[src], h[dst])

        # 计算损失函数的值
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        # print('neg_loss:', neg_loss)
        # print('pos_loss:', pos_loss, 'neg_loss:', neg_loss)
        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        # gradient quantization
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
    print('dst_vertex_num_avg:', dst_vertex_num / num_i)

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size, cluster_label=None):
    model.eval()
    predictor.eval()

    h = model(x, adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        if args.use_cluster:
            src = cluster_label[edge[0]]
            dst = cluster_label[edge[1]]
        else:
            src = edge[0]
            dst = edge[1]
        pos_train_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        if args.use_cluster:
            src = cluster_label[edge[0]]
            dst = cluster_label[edge[1]]
        else:
            src = edge[0]
            dst = edge[1]
        pos_valid_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        if args.use_cluster:
            src = cluster_label[edge[0]]
            dst = cluster_label[edge[1]]
        else:
            src = edge[0]
            dst = edge[1]
        neg_valid_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        if args.use_cluster:
            src = cluster_label[edge[0]]
            dst = cluster_label[edge[1]]
        else:
            src = edge[0]
            dst = edge[1]
        pos_test_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        if args.use_cluster:
            src = cluster_label[edge[0]]
            dst = cluster_label[edge[1]]
        else:
            src = edge[0]
            dst = edge[1]
        neg_test_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results
