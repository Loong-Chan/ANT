import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import itertools
from torch import optim
from sklearn.mixture import GaussianMixture
from Model.Encoder import Encoder
from Model.Decoder import Decoder


class ANT(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, 
                 latent_cluster, conditional_size, T, device):
        super().__init__()

        self.temp = T
        self.latent_cluster = latent_cluster
        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional_size)

        self.mu_c = nn.Parameter(torch.FloatTensor(latent_cluster, latent_size))
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(latent_cluster, latent_size))

        self.environment = None
        self.learnable_meta_pi = None
        self.device = device
    

    def forward(self, x, idx_train, labels):
        mu_z, sigma2_log_z = self.encoder(x, self.environment)
        z = self.reparameterize(mu_z, sigma2_log_z)
        recon_x = self.decoder(mu_z, self.environment)
        recon_loss = ((recon_x - x)**2).sum()

        ali_loss = self.ALI_loss(z, labels, idx_train)

        kl_loss = self.KLD_loss(z, mu_z, sigma2_log_z)

        return recon_loss / x.size(0), kl_loss / x.size(0), ali_loss / idx_train.size(0), \
               recon_x, mu_z.squeeze(1), sigma2_log_z.squeeze(1), z


    def ALI_loss(self, z, labels, idx_train):
        log_cluster_prob = gmm_proba(z, self.mu_c, self.log_sigma2_c, use_exp=False)
        train_log_cluster_prob = log_cluster_prob[idx_train]
        train_labels = labels[idx_train]

        class_to_cluster = []
        for i in range(labels.max().item()+1):
            class_to_cluster.append(torch.mean(train_log_cluster_prob[train_labels == i], dim=0))
        class_to_cluster = torch.cat(class_to_cluster).reshape(-1, labels.max().item()+1)
        match = linear_sum_assignment(-class_to_cluster.detach().cpu())[1]

        matched_train_log_cluster_prob = train_log_cluster_prob.T[match].T
        ali_loss = F.nll_loss(matched_train_log_cluster_prob, train_labels)
        return ali_loss


    def KLD_loss(self, z, mu_z, sigma2_log_z):
        cluster_prob = gmm_proba(z, self.mu_c, self.log_sigma2_c, T=self.temp)
        meta_pi = F.softmax(self.learnable_meta_pi, dim=1)
        pi = torch.mm(self.environment, meta_pi)
        gamma_c = cluster_prob * pi
        gamma_c = gamma_c / gamma_c.sum(dim=1, keepdim=True)
        log_sigma2_c = self.log_sigma2_c.unsqueeze(0)  # [1, latent_cluster, z_dim]
        mu_c = self.mu_c.unsqueeze(0)  # [1, latent_cluster, z_dim]
        log_sigma2_z = sigma2_log_z.unsqueeze(1)  # [batch_size, 1, z_dim]
        mu_z = mu_z.unsqueeze(1)  # [batch_size, 1, z_dim]
        kl_loss_1 = log_sigma2_c + torch.exp(log_sigma2_z - log_sigma2_c) + (mu_z - mu_c).pow(2) / torch.exp(log_sigma2_c)
        kl_loss_1 = 0.5 * torch.sum(gamma_c * torch.sum(kl_loss_1, 2))
        kl_loss_2 = torch.sum(gamma_c * torch.log(pi / gamma_c))
        kl_loss_3 = 0.5 * torch.sum(1 + log_sigma2_z)
        kl_loss = kl_loss_1 - kl_loss_2 - kl_loss_3
        return kl_loss


    def reparameterize(self, means, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return means + eps * std


    def pretrain(self, features, environment, pretrain_epochs, pretrain_lr, verbose):
        
        self.environment = environment.to(torch.float32)

        pretrain_optim = optim.Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()), lr=pretrain_lr)
        for epoch in range(pretrain_epochs):
            mu_z, sigma2_log_z = self.encoder(features, self.environment)
            z = self.reparameterize(mu_z, sigma2_log_z)
            recon_feat = self.decoder(z, self.environment)       
            recon_loss = ((recon_feat - features)**2).sum()
            kl_loss = -0.5 * torch.sum(1 + sigma2_log_z - mu_z.pow(2) - sigma2_log_z.exp())
            pretrain_loss = recon_loss + 2 * kl_loss
            pretrain_optim.zero_grad()
            pretrain_loss.backward()
            pretrain_optim.step()
            if verbose:
                print(f"[pretrain] Epoch:{epoch+1:3d}/{pretrain_epochs}, Pretrain Loss:{pretrain_loss.detach().item():.4f}")   

        Z = []
        with torch.no_grad():
            mu_z, sigma2_log_z = self.encoder(features, self.environment)  # envirment
            Z.append(mu_z)
        Z = torch.cat(Z, 0).detach()
        cluster = GaussianMixture(n_components=self.latent_cluster, covariance_type='diag')
        cluster.fit_predict(torch.unique(Z, dim=0).cpu())
        self.mu_c.data = torch.from_numpy(cluster.means_).float().to(self.device)
        self.log_sigma2_c.data = torch.log(torch.from_numpy(cluster.covariances_).float()).to(self.device)
        cls_prob = cluster.predict_proba(Z.cpu())

        pi = torch.mm(self.environment.T, torch.FloatTensor(cls_prob).to(self.device))
        pi = tensor_rownorm(pi)
        self.learnable_meta_pi = nn.Parameter(pi)



def gmm_proba(x, mus, log_sigma2s, eps=1e-10, T=1e2, use_exp=True):
    expand_x = x.repeat([1, mus.shape[0]]).reshape([-1, x.shape[1]])
    expand_mus = mus.repeat([x.shape[0], 1])
    expand_log_sigma2s = log_sigma2s.repeat([x.shape[0], 1])
    log_prob = gaussian_pdf_log(expand_x, expand_mus, expand_log_sigma2s).reshape([-1, mus.shape[0]])
    if use_exp:
        max_log_prob = log_prob.max().detach()
        prob = torch.exp((log_prob - max_log_prob) / T) + eps
        return prob / prob.sum(dim=1, keepdim=True)
    else:
        return log_prob



def gaussian_pdf_log(x, mu, log_sigma2):
    return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))



def tensor_rownorm(tensor):
    r_inv = tensor.sum(dim=1).pow(-1)
    r_inv[r_inv == float("Inf")] = 0.
    r_inv[r_inv == float("-Inf")] = 0.
    return tensor * r_inv.reshape(-1, 1)
