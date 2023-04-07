from .global_variable import run_recorder, args
from .other import dec2bin
from models import NamedGCNConv
from models import WAGERounding, C
import numpy as np
import torch.nn as nn
import os
import torch

current_layer = -1
current_epoch = 1
matrix_activity = 0
vertex_map = None
updated_vertex_map = None


def hook_set_epoch(self: NamedGCNConv, input_data):
    global current_epoch
    current_epoch = self.get_epoch()
    if current_epoch == 1:
        global matrix_activity
        matrix_activity = self.adj_activity


def hook_forward_set_grad_zero(module, input_data, output_data):
    assert updated_vertex_map is not None
    if vertex_map is not None:
        for i, vertex_index in enumerate(vertex_map):
            if updated_vertex_map[i] == 0:
                output_data[vertex_index, :] = input_data[0][vertex_index, :]
    else:
        for i, is_updated in enumerate(updated_vertex_map):
            if is_updated == 0:
                output_data[i, :] = input_data[0][i, :]
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
    input_binary = np.zeros([input_data[0].shape[1], input_data[0].shape[0] * args.bl_activate], dtype=np.str)
    input_binary_col, scale = dec2bin(input_data[0].data.cpu().data.numpy(), args.bl_activate)
    for i, b in enumerate(input_binary_col):
        input_binary[:, i::args.bl_activate] = b.transpose()
    activity = np.sum(input_binary.astype(np.float), axis=None) / np.size(input_binary)
    input_c = run_recorder.record(f'layer_run/epoch{current_epoch}', f'convs.{layer}.gcn_conv.lin.input_C.csv',
                                  input_binary,
                                  delimiter=',', fmt='%s')
    f = open(run_recorder.bootstrap_path, 'a')
    weight_c = input_c.replace('input_C', 'weight_before')
    weight_updated_c = weight_c.replace('before', 'after')
    f.write(weight_updated_c + ' ' + weight_c + ' ' + input_c + ' ' + str(activity) + ' ')
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
    f.write(weight_updated_a + ' ' + weight_a + ' ' + input_a + ' ' + str(matrix_activity) + ' ')
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
