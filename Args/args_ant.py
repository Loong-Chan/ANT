import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--VAE_hidden_size", type=int, default=256)
parser.add_argument("--VAE_z_size", type=int, default=8)
parser.add_argument("--temperature", type=float, default=5000)
parser.add_argument("--device_idx", default="0")
parser.add_argument("--dataset", default="photo")
parser.add_argument("--pretrain_epoch", type=int, default=2000)
parser.add_argument("--pretrain_lr", type=float, default=0.002)
parser.add_argument("--verbose", default=True)
parser.add_argument("--ant_lr", type=float, default=5e-4)
parser.add_argument("--ant_epoch", type=int, default=2000)
parser.add_argument("--beta", type=float, default=10.0)


ANT_args = parser.parse_args()