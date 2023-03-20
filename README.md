# GNN_PIM_Demo
it is just a original demo!

#### 1.使用说明

1. 启动：
   1. 查看 `util/global_variable.py` 中可输入的参数列表；
   2. 输入`python train_gnn_ddi.py 各类参数`开始模拟，每个执行一个epoch会自动调用Neurosim内核进行GCN-PIM加速器性能及能耗评估；
2. 修改：若希望自定义可修改以下文件，修改后需要在`NeuroSIM/`下重新执行`make`命令；
   1. `NeuroSIM/Network.csv`：修改NeuroSim能够感知到的网络结构；
   2. `NeuroSIM/Param.cpp`：修改NeuroSim模拟的GCN-PIM加速器的各类硬件参数；



#### 2.文件夹及文件内容

- `models/`
  - `gat.py`：GAT模型实现类；
  - `gcn.py`：GCN模型实现类；
  - `graphsage.py`：GraphSAGE模型实现类；
  - `linkpredictor.py`：ddi数据集预测边（代表两个药物是否存在共同作用）时使用的预测器的实现类；
  - `named_gcnconv.py`：PYG框架中 `gcn_conv` 类的包装类，便于添加自定义属性；
  - `vgg_low.py`：VGG模型实现类；
  - `wage_initializer.py`：包含WAGE量化框架与初始化有关的函数；
  - `wage_quantizer.py`：包含WAGE量化框架的各类量化函数；
- `util/`
  - `global_variable.py`：用于声明各类全局变量；
  - `hook.py`：包含神经网络训练过程中使用的各类钩子函数，用于获取各层前向传播或反向传播的结果，并通过下面的`recorder`类对象保存为文件；
  - `logger.py`：用于打印输出到控制台的消息的类；
  - `other.py`：记录其它的一些工具函数，目前包括稀疏矩阵正则化函数和二进制转换函数；
  - `recorder.py`：用于将数据保存为指定格式的文件；
  - `train_decorator.py`：神经网络训练的装饰器类，方便在训练过程中进行：添加hook函数，量化权重/梯度，创建启动NeuroSim的命令文件（.sh）等操作；
  - `train_test_ddi.py`：包含使用ddi数据集进行边预测时用到的训练和测试函数；
- `NeuroSim_Results_Each_Epoch/`（**运行后生成**）：记录开始GCN-NeuroSim模拟后，每个epoch，NeuroSim的模拟结果；
- `record/`（**运行后生成**）：记录模型训练过程中python部分生成的文件，作为NeuroSim模拟的输入，包含各层的输入、更新前后的权重、邻接矩阵、NeuroSim启动命令等；
- `train_gnn_arxiv.py`：利用arxiv数据集进行图神经网络训练的入口文件（主文件），**目前还在调试中**；
- `train_gnn_ddi.py`：利用ddi数据集进行图神经网络训练的入口文件（主文件），**现在主要用这个**；
