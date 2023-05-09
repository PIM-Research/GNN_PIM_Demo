import networkx as nx
import random
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
import torch
import torch_geometric.transforms as T
import scipy.sparse as sp
import networkx as nx
from torch_sparse import SparseTensor
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


def random_walk(graph, start_node, length):
    walk = [start_node]
    for _ in range(length - 1):
        neighbors = list(graph.neighbors(walk[-1]))
        if neighbors:
            walk.append(random.choice(neighbors))
        else:
            break
    return walk


# dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
# dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
dataset = PygLinkPropPredDataset(name='ogbl-ppa', transform=T.ToSparseTensor())
data = dataset[0]
adj = data.adj_t
# 将SparseTensor类型转换成COO稀疏矩阵的形式
coo = adj.coo()
print(coo[0].shape, coo[1].shape)
value = torch.ones(size=coo[0].shape)
# 将COO稀疏矩阵转换成scipy.sparse中的稀疏矩阵类型
sparse_matrix = sp.coo_matrix((value.tolist(), (coo[0].numpy(), coo[1].numpy())))
# 将scipy.sparse中的稀疏矩阵类型转换成networkx中的图类型
graph = nx.from_scipy_sparse_array(sparse_matrix)

# 执行随机游走
num_walks = 10  # 游走次数
walk_length = 5  # 游走长度
walks = []
for _ in range(num_walks):
    for node in graph.nodes():
        walk = random_walk(graph, node, walk_length)
        walks.append(walk)

sentences = [list(map(str, walk)) for walk in walks]  # 将序列转换为字符串列表
model = Word2Vec(vector_size=64, window=5, sg=1, hs=0, negative=10, alpha=0.03, min_alpha=0, seed=14)  # 训练Word2Vec模型
model.build_vocab(walks, progress_per=2)
model.train(walks, total_examples=model.corpus_count, epochs=50, report_delay=1)
# 获取节点的嵌入向量
node_embeddings = model.wv.vectors
print(type(node_embeddings))
print(node_embeddings.size)
