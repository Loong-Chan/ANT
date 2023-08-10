import os
import torch
from make_dataset import processed_dataset
from Model.ANT import ANT 
from Args.args_ant import ANT_args
from torch import optim
import utils


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ANT_args.device_idx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


adj, features, labels, idx_train, idx_val, idx_test, environment, edge_index = processed_dataset(ANT_args.dataset, device)


utils.set_seed(ANT_args.seed)
model = ANT(encoder_layer_sizes=[features.shape[1], ANT_args.VAE_hidden_size],
            latent_size=ANT_args.VAE_z_size,
            decoder_layer_sizes=[ANT_args.VAE_hidden_size, features.shape[1]],
            latent_cluster=labels.max().item()+1,
            conditional_size=environment.shape[1],
            T=ANT_args.temperature,
            device=device)
model.to(device)


model.pretrain(features, environment=environment, 
               pretrain_epochs=ANT_args.pretrain_epoch, 
               pretrain_lr=ANT_args.pretrain_lr, verbose=ANT_args.verbose)



optimizer = optim.Adam(model.parameters(), lr=ANT_args.ant_lr)
for e in range(ANT_args.ant_epoch):
    model.train()
    recon_loss, kl_loss, sup_loss, recon_x, mean, log_var, z = model(features, idx_train, labels)
    optimizer.zero_grad()
    loss = recon_loss + ANT_args.beta * kl_loss + sup_loss
    loss.backward()
    optimizer.step()
    if ANT_args.verbose:
        print(f"[train] Epoch:{e+1:3d}/{ANT_args.ant_epoch}, Loss:{loss.detach().item():.4f}") 


model.eval()
model.environment = environment.mean(dim=0).repeat([adj.shape[0], 1]).to(device)


torch.save(model, f"Tmp/ANT_{ANT_args.dataset}.pth")
