from util.hook import hook_Layer_output, hook_combination_input_output, hook_set_epoch
import os


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
            if 'weight' in name:
                param.data = self.weight_quantification(model.weight_acc[name], model.weight_scale[name]).to(
                    next(model.parameters()).device)
                if self.recorder is not None and batch_index == 0:
                    self.recorder.record(f'layer_run/epoch{cur_epoch}', f'{name}_before.csv',
                                         param.data.T.to('cpu').data.numpy(),
                                         delimiter=',', fmt='%10.5f')

    def bind_hooks(self, model, batch_index, cur_epoch):
        if batch_index == 0:
            for index, (name, layer) in enumerate(model.convs.named_children()):
                layer.set_epoch(cur_epoch)
                if index == 0:
                    self.hook_handle_list.append(layer.register_forward_pre_hook(hook_set_epoch))
                for index_c, (name_c, layer_c) in enumerate(layer.gcn_conv.named_children()):
                    self.hook_handle_list.append(layer_c.register_forward_hook(hook_combination_input_output))
                # self.hook_handle_list.append(layer.register_forward_hook(hook_Layer_output))

    def quantify_activation(self, model):
        for name, param in list(model.named_parameters())[::-1]:
            if 'weight' in name:
                param.grad.data = self.grad_quantification(param.grad.data).data
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
                if 'weight' in name and self.recorder is not None:
                    self.recorder.record(f'layer_run/epoch{cur_epoch}', f'{name}_after.csv',
                                         model.weight_acc[name].T.to('cpu').data.numpy(),
                                         delimiter=',', fmt='%10.5f')
