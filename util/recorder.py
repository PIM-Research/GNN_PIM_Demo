import os
import numpy as np


class Recorder:
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.bootstrap_path = os.path.join(self.dir_name, 'trace_command.sh')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def record(self, label, file_name, data, delimiter=',', fmt='%10f'):
        label_dir = os.path.join(self.dir_name, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        file_path_ab = os.path.join(label_dir, file_name)
        np.savetxt(file_path_ab, data, delimiter=delimiter, fmt=fmt)
        return file_path_ab

    def record_change(self, label, file_name, data_before, data_after, delimiter=',', fmt='%10f'):
        file_path_ab_before = self.record(label, f'{file_name}_before.csv', data_before, delimiter, fmt)
        file_path_ab_after = self.record(label, f'{file_name}_after.csv', data_after, delimiter, fmt)
        return file_path_ab_before, file_path_ab_after

    def record_acc_vertex_map(self, label, file_name, data: np.ndarray, vertex_map: np.ndarray, delimiter=',',
                              fmt='%10f'):
        assert data.shape[0] == vertex_map.shape[0]
        data_mapped = np.zeros(shape=data.shape)
        for i, pointer in enumerate(vertex_map):
            data_mapped[i] = data[pointer]
        return self.record(label, file_name, data_mapped, delimiter, fmt)
