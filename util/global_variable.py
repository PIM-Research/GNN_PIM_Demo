import argparse
import time
from util import recorder

# 实例化recorder,parser
create_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
dir_name = f'./record/ddi/{create_time}'
run_recorder = recorder.Recorder(dir_name)
parser = argparse.ArgumentParser(description='OGBN-ddi (GNN)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--use_sage', action='store_true')
parser.add_argument('--use_gcn', action='store_false')
parser.add_argument('--batch_size', type=int, default=64 * 1024)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--bl-weight', type=int, default=5, metavar='N',
                    help='word length in bits for weight output; -1 if full precision.')
parser.add_argument('--bl-grad', type=int, default=5, metavar='N',
                    help='word length in bits for gradient; -1 if full precision.')
parser.add_argument('--bl-activate', type=int, default=8, metavar='N',
                    help='word length in bits for layer activations; -1 if full precision.')
parser.add_argument('--bl-error', type=int, default=8, metavar='N',
                    help='word length in bits for backward error; -1 if full precision.')
parser.add_argument('--bl-rand', type=int, default=16, metavar='N',
                    help='word length in bits for rand number; -1 if full precision.')
args = parser.parse_args()
