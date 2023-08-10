import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device_idx", default="0")
parser.add_argument("--dataset", default="photo")
parser.add_argument("--backbone", default="GCNII")
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--n_cat", type=int, default=2)
parser.add_argument("--hidden", type=int, default=16)
parser.add_argument("--consis", default=False)
parser.add_argument("--n_aug", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--gnn_epoch", type=int, default=1000)
parser.add_argument("--weight_decay", type=float, default=1e-5)

parser.add_argument("--verbose", default=True)

GNN_args = parser.parse_args()