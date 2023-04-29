import torch

from util.hook import hook_combination_input_output, hook_set_epoch, hook_forward_set_grad_zero
import os

from util.other import filter_edges


class TrainDecorator:
    def __init__(self, weight_quantification, grad_quantification, wage_grad_clip, recorder):
        self.hook_handle_list = []
        self.weight_quantification = weight_quantification
        self.grad_quantification = grad_quantification
        self.wage_grad_clip = wage_grad_clip
        self.recorder = recorder

    def create_bash_command(self, num_epoch, bl_weight, bl_activation):
        sim_bootstrap = self.recorder.bootstrap_path
        if os.path.exists(sim_bootstrap):
            os.remove(sim_bootstrap)
        f = open(sim_bootstrap, "w")
        f.write('./NeuroSIM/main ' + str(num_epoch) + ' ./NeuroSIM/NetWork.csv ' + str(bl_weight) + ' ' + str(
            bl_activation) + ' ')

    def quantify_weight(self, model, batch_index, cur_epoch):
        for name, param in model.named_parameters():
            if 'weight' in name and 'convs' in name:
                if self.recorder is not None and batch_index == 0:
                    self.recorder.record(f'layer_run/epoch{cur_epoch}', f'{name}_before.csv',
                                         param.data.T.to('cpu').data.numpy(),
                                         delimiter=',', fmt='%10.5f')
                param.data = self.weight_quantification(model.weight_acc[name], model.weight_scale[name]).to(
                    next(model.parameters()).device)
                # if self.recorder is not None and batch_index == 0:
                #     self.recorder.record(f'layer_run/epoch{cur_epoch}', f'{name}_before.csv',
                #                          param.data.T.to('cpu').data.numpy(),
                #                          delimiter=',', fmt='%10.5f')

    def bind_hooks(self, model, batch_index, cur_epoch):
        if batch_index == 0:
            for index, (name, layer) in enumerate(model.convs.named_children()):
                layer.set_epoch(cur_epoch)
                if index == 0:
                    self.hook_handle_list.append(layer.register_forward_pre_hook(hook_set_epoch))
                for index_c, (name_c, layer_c) in enumerate(layer.gcn_conv.named_children()):
                    self.hook_handle_list.append(layer_c.register_forward_hook(hook_combination_input_output))
                # self.hook_handle_list.append(layer.register_forward_hook(hook_Layer_output))

    def quantify_grad(self, model):
        for name, param in list(model.named_parameters())[::-1]:
            if 'weight' in name and 'convs' in name:
                # print('before:', param.grad.data)
                param.grad.data = self.grad_quantification(param.grad.data).data
                # print('after:', param.grad.data)
                # 裁剪，限制权重范围
                w_acc = self.wage_grad_clip(model.weight_acc[name]).to(next(model.parameters()).device)
                w_acc -= param.grad.data
                model.weight_acc[name] = w_acc

    def clear_hooks(self, model, batch_index, cur_epoch):
        if batch_index == 0:
            f = open(self.recorder.bootstrap_path, 'a')
            f.write(os.path.join(self.recorder.dir_name, 'updated_vertex.csv'))
            for handle in self.hook_handle_list:
                handle.remove()
            for name, param in list(model.named_parameters())[::-1]:
                if 'weight' in name and self.recorder is not None and 'convs' in name:
                    self.recorder.record(f'layer_run/epoch{cur_epoch}', f'{name}_after.csv',
                                         param.data.T.to('cpu').data.numpy(),
                                         delimiter=',', fmt='%10.5f')
                    # self.recorder.record(f'layer_run/epoch{cur_epoch}', f'{name}_after.csv',
                    #                      model.weight_acc[name].T.to('cpu').data.numpy(),
                    #                      delimiter=',', fmt='%10.5f')

    @staticmethod
    def bind_update_hook(model):
        # 添加钩子使得drop掉的顶点特征不更新
        for index, (name, layer) in enumerate(model.convs.named_children()):
            for index_c, (name_c, layer_c) in enumerate(layer.gcn_conv.named_children()):
                layer_c.register_forward_hook(hook_forward_set_grad_zero)

    @staticmethod
    def filter_adj_by_batch(adj_t, source_vertexes, dst_vertexes, batch_index):
        vertex_filter = torch.unique(torch.cat((source_vertexes, dst_vertexes), dim=-1))
        adj_t = filter_edges(adj_t, vertex_filter)
        return adj_t.to_device(adj_t.device())
