import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Experiment Runner')
parser.add_argument('--script_path', metavar='S', help='path to train script', default='train.py')
parser.add_argument('--lr_max', default=1e-3, type=float, help='lr max point')
parser.add_argument('--lr_min', default=1e-6, type=float, help='lr min point')
parser.add_argument('--lr_jump', default=1e-6, type=float, help='lr start point')
args = parser.parse_args()

lr_list = list(np.arange(args.lr_min, args.lr_max, args.lr_jump))
print "===> Starting grid search"
print "===> LR:", lr_list

for lr in lr_list:
    print "===> Running with LR=%f" % lr