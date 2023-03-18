import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from models import GCN
from .train_decorator import TrainDecorator


def train(model: GCN, predictor, x, adj_t, split_edge, optimizer, batch_size, train_decorator: TrainDecorator,
          cur_epoch=0):
    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    train_decorator.create_bash_command(cur_epoch, model.bits_W, model.bits_A)
    model.train()
    predictor.train()
    # 获取训练集的边集，即图中存在的所有边的集合，类型是tensor，共两维，行数代表边数，列为源节点和目标节点
    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    # perm实际上是一个子图，类型是tensor，元素代表本次预测用到的边的编号，这里perm起到索引作用
    # 这个数据集边的数量为1067911，batch_size为65536，所以有17个batch
    for i, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size,
                                        shuffle=True)):
        # 量化权重
        train_decorator.quantify_weight(model, i, cur_epoch)

        # 绑定钩子函数，记录各层的输入
        train_decorator.bind_hooks(model, i, cur_epoch)

        optimizer.zero_grad()

        # 进行图卷积计算
        h = model(x, adj_t)

        # 根据perm这个索引获得本次预测需要的边
        edge = pos_train_edge[perm].t()

        # 预测这两个顶点之间是否存在边，1代表存在，0为不存在
        pos_out = predictor(h[edge[0]], h[edge[1]])
        # 计算损失函数的值
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # 什么是负采样？
        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        # 预测这两个顶点之间是否存在边，1代表存在，0为不存在
        neg_out = predictor(h[edge[0]], h[edge[1]])
        # 计算损失函数的值
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        # gradient quantization
        # 量化梯度
        train_decorator.quantify_activation(model)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        # 清楚钩子
        train_decorator.clear_hooks(model, i, cur_epoch)

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size):
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
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
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
