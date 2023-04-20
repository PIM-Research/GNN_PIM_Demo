import argparse
import time

from models import QW, QG, C
from util import recorder
import os

parser = argparse.ArgumentParser(description='OGBN-ddi (GNN)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--percentile', type=int, default=50)
parser.add_argument('--array-size', type=int, default=64)
parser.add_argument('--drop-mode', type=int, default=1)
parser.add_argument('--log-steps', type=int, default=1)
parser.add_argument('--eval-steps', type=int, default=1)
parser.add_argument('--eps', type=float, default=0.01)
parser.add_argument('--min-samples', type=int, default=3)
parser.add_argument('--kmeans-clusters', type=float, default=0.5)
parser.add_argument('--n-init', type=int, default=10)
parser.add_argument('--cluster-alg', type=int, default=1)
parser.add_argument('--cluster-basis', type=int, default=1)
parser.add_argument('--negs', type=int, default=1)
parser.add_argument('--mapping-alg', type=int, default=1)
parser.add_argument('--use-sage', action='store_true')
parser.add_argument('--call-neurosim', action='store_true')
parser.add_argument('--add-self-loop', action='store_true')
parser.add_argument('--use-gcn', action='store_false')
parser.add_argument('--use-cluster', action='store_true')
parser.add_argument('--batch-size', type=int, default=64 * 1024)
parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--hidden-channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--record-dir', type=str, default='ddi')
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

# 实例化recorder
create_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
dir_name = f'./record/{args.record_dir}/{create_time}'
# 实例化parser
run_recorder = recorder.Recorder(dir_name)

if not os.path.exists('./NeuroSim_Results_Each_Epoch'):
    os.makedirs('./NeuroSim_Results_Each_Epoch')
if not os.path.exists('./result'):
    os.makedirs('./result')
result_file_path = f'./result/{create_time}.txt'
with open(result_file_path, 'w') as f:
    f.write(f'{str(args)}\n')

weight_quantification = lambda x, scale: QW(x, args.bl_weight, scale)
grad_quantiication = lambda x: QG(x, args.bl_grad, args.bl_rand, args.lr)
grad_clip = lambda x: C(x, args.bl_weight)