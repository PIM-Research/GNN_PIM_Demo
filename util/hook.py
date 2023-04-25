from .global_variable import run_recorder, args
from models import NamedGCNConv
from models import WAGERounding, C
import torch.nn as nn
import os
import torch
from torch_sparse import SparseTensor

current_layer = -1
current_epoch = 1
matrix_activity = 0
vertex_map = None
updated_vertex_map = None
min_dis_vertex = 0


def hook_set_epoch(self: NamedGCNConv, input_data):
    global current_epoch
    current_epoch = self.get_epoch()
    if current_epoch == 1:
        global matrix_activity
        matrix_activity = self.adj_activity


# def hook_forward_set_grad_zero(module, input_data, output_data):
#     assert updated_vertex_map is not None
#     if input_data[0].shape[1] == output_data.shape[1]:
#         if vertex_map is not None:
#             for i, vertex_index in enumerate(vertex_map):
#                 if updated_vertex_map[i] == 0:
#                     output_data[vertex_index, :] = input_data[0][vertex_index, :]
#         else:
#             for i, is_updated in enumerate(updated_vertex_map):
#                 if is_updated == 0:
#                     output_data[i, :] = input_data[0][i, :]
#     return output_data


def hook_forward_set_grad_zero(module, input_data, output_data):
    assert updated_vertex_map is not None
    if input_data[0].shape[1] == output_data.shape[1]:
        mask = vertex_map[updated_vertex_map == 0] if vertex_map is not None else updated_vertex_map == 0
        output_data[mask, :] = input_data[0][mask, :]
    return output_data


def hook_Layer_output(self: NamedGCNConv, input_data, output_data):
    weight_updated_a = run_recorder.record(f'layer_run/epoch{current_epoch}', f'{self.name}.output.csv',
                                           output_data.data.to('cpu').data.numpy(),
                                           delimiter=',', fmt='%10.5f')
    f = open(run_recorder.bootstrap_path, 'a')
    weight_a = weight_updated_a.replace('output.csv', 'lin.output_C.csv')
    input_a = os.path.join(run_recorder.dir_name, 'adj_matrix.csv')
    f.write(weight_updated_a + ' ' + weight_a + ' ' + input_a + ' ' + str(self.adj_activity) + ' ')
    f.close()


def hook_combination_input_output(self: nn.Linear, input_data, output_data):
    layer = get_current_layer()
    # input_coo = SparseTensor.from_dense(output_data.detach()[min_dis_vertex].t()).coo()
    # print(input_data)
    input_vertex = input_data[0].detach()[min_dis_vertex].cpu()
    row, col = torch.arange(0, input_vertex.shape[0], dtype=torch.long), torch.zeros(size=input_vertex.shape,
                                                                                     dtype=torch.long)
    input_stack = torch.stack([row, col, input_vertex]).to('cpu').numpy()
    input_c = run_recorder.record(f'layer_run/epoch{current_epoch}', f'convs.{layer}.gcn_conv.lin.input_C.csv',
                                  input_stack, delimiter=',', fmt='%s')
    f = open(run_recorder.bootstrap_path, 'a')
    weight_c = input_c.replace('input_C', 'weight_before')
    weight_updated_c = weight_c.replace('before', 'after')
    f.write(weight_updated_c + ' ' + weight_c + ' ' + input_c + ' 0 ')
    # f.close()
    # 这里对Combination的输出进行量化
    if args.bl_activate != -1:
        output_data = C(output_data, args.bl_activate)  # keeps the gradients
    output_data = WAGERounding.apply(output_data, args.bl_activate, args.bl_error, None)

    # 判断drop模式是否是GLOBAL
    if vertex_map is not None:
        weight_a = run_recorder.record_acc_vertex_map(f'layer_run/epoch{current_epoch}',
                                                      f'convs.{layer}.gcn_conv.lin.output_C.csv',
                                                      output_data.data.to('cpu').data.numpy(), vertex_map,
                                                      delimiter=',', fmt='%10.5f')
    else:
        weight_a = run_recorder.record(f'layer_run/epoch{current_epoch}', f'convs.{layer}.gcn_conv.lin.output_C.csv',
                                       output_data.data.to('cpu').data.numpy(), delimiter=',', fmt='%10.5f')

    # weight_updated_a = f'convs.{layer - 1}.gcn_conv.lin.output_C.csv'
    weight_updated_a = os.path.join(run_recorder.dir_name, f'layer_run/epoch{current_epoch}',
                                    f'convs.{layer - 1}.gcn_conv.lin.output_C.csv')
    if layer == 0:
        row_num = output_data.data.shape[0]
        col_num = output_data.data.shape[1]
        run_recorder.record(f'layer_run/epoch{current_epoch}', f'convs.{layer - 1}.gcn_conv.lin.output_C.csv',
                            torch.zeros([row_num, col_num]).data.numpy(),
                            delimiter=',', fmt='%10.5f')
    input_a = os.path.join(run_recorder.dir_name, 'adj_matrix.csv')
    f.write(weight_updated_a + ' ' + weight_a + ' ' + input_a + ' 0 ')
    f.close()


def get_current_layer():
    global current_layer
    current_layer += 1
    if current_layer == args.num_layers:
        current_layer = 0
    return current_layer


def set_vertex_map(vertex_pointer):
    global vertex_map
    vertex_map = vertex_pointer


def set_updated_vertex_map(updated_vertex):
    global updated_vertex_map
    updated_vertex_map = updated_vertex


def set_min_dis_vertex(index):
    global min_dis_vertex
    min_dis_vertex = index
