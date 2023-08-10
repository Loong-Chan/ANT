import os
import torch
from make_dataset import processed_dataset
from Args.args_gnn import GNN_args
from Backbone import GCN, GAT, MLP, GCNII, S2GC, GATv2
from torch import optim
import torch.nn.functional as F
import utils

utils.set_seed(GNN_args.seed)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GNN_args.device_idx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


adj, features, labels, idx_train, idx_val, idx_test, environment, edge_index = processed_dataset(GNN_args.dataset, device)
model = torch.load(f"Tmp/ANT_{GNN_args.dataset}.pth").to(device)


def get_aug_x():
    feature_list = [features]
    for _ in range(GNN_args.n_cat):
        feature_list.append(model(features, idx_train, labels)[3].detach())
    return torch.cat(feature_list, dim=1)


nfeat = get_aug_x().shape[1]


backbone = GNN_args.backbone
gnn_hidden_size = GNN_args.hidden
gnn_lr = GNN_args.lr
gnn_weight_decay = GNN_args.weight_decay


if backbone == "GCN":
    GNN = GCN(nfeat=nfeat, nhid=gnn_hidden_size, 
                nclass=labels.max().item()+1, dropout=0.5)
    GNN_optim = optim.Adam(GNN.parameters(), lr=gnn_lr, weight_decay=gnn_weight_decay)
if backbone == "GAT":
    GNN = GAT(nnode=features.shape[0], nfeat=nfeat, nhid=gnn_hidden_size, 
                nclass=labels.max().item()+1, dropout=0.5, nheads=3, alpha=0.1)
    GNN_optim = optim.Adam(GNN.parameters(), lr=gnn_lr, weight_decay=gnn_weight_decay)
if backbone == "GATv2":
    GNN = GATv2(nnode=features.shape[0], nfeat=nfeat, nhid=gnn_hidden_size, 
                nclass=labels.max().item()+1, dropout=0.5, nheads=3, alpha=0.1)
    GNN_optim = optim.Adam(GNN.parameters(), lr=gnn_lr, weight_decay=gnn_weight_decay)
if backbone == "MLP":
    GNN = MLP(nfeat=nfeat, nhid=gnn_hidden_size,
                nclass=labels.max().item()+1, nlayer=2)
    GNN_optim = optim.Adam(GNN.parameters(), lr=gnn_lr, weight_decay=gnn_weight_decay)
if backbone == "GCNII":
    GNN = GCNII(nfeat=nfeat, nlayers=64,
                nhidden=64, nclass=labels.max().item()+1, 
                dropout=0.6, lamda=0.5, alpha=0.1, variant=False)
    GNN_optim = optim.Adam([
                            {'params':GNN.params1,'weight_decay':0.01},
                            {'params':GNN.params2,'weight_decay':gnn_weight_decay},
                            ],lr=gnn_lr)
if backbone == "S2GC":
    GNN = S2GC(nfeat=nfeat, nclass=labels.max().item()+1,
                features=get_aug_x().clone(), adj=adj.clone(), degree=16, alpha=0.05)
    GNN_optim = optim.Adam(GNN.parameters(), lr=gnn_lr, weight_decay=gnn_weight_decay)


GNN.to(device)


for epoch in range(GNN_args.gnn_epoch):
    GNN.train()
    GNN_optim.zero_grad()
    loss_train = 0.
    output_list = []
    for s in range(GNN_args.n_aug):
        output = GNN(x=get_aug_x(), adj=adj, edge_index=edge_index)
        loss_train += F.nll_loss(output[idx_train], labels[idx_train])
        output_list.append(output)
    loss = loss_train / GNN_args.n_aug
    if GNN_args.consis:
        loss += utils.consis_loss(output_list, 0.5)
    loss.backward()
    GNN_optim.step()
    GNN.eval()
    output = GNN(x=get_aug_x(), adj=adj, edge_index=edge_index)
    acc_val = utils.accuracy(output[idx_val], labels[idx_val])
    acc_test = utils.accuracy(output[idx_test], labels[idx_test])
    if GNN_args.verbose:
        print(f"[GNN] Epoch:{epoch+1:3d}/{GNN_args.gnn_epoch}, Train Loss:{loss.detach().item():.4f} ",
              f"Acc Val:{acc_val:.4f}, ACC Test:{acc_test:.4f}") 

GNN.eval()
output = GNN(x=get_aug_x(), adj=adj, edge_index=edge_index)
acc_test = utils.accuracy(output[idx_test], labels[idx_test])
print(f"Acc: {acc_test}")


