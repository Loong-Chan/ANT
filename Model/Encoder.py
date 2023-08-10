import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional_size):
        super().__init__()
        local_layer_size = layer_sizes.copy()
        local_layer_size[0] += conditional_size
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(local_layer_size[:-1], local_layer_size[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        self.linear_means = nn.Linear(local_layer_size[-1], latent_size)
        self.linear_log_var2 = nn.Linear(local_layer_size[-1], latent_size)

    def forward(self, x, c):
        if c is not None:
            x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        log_var2 = self.linear_log_var2(x)
        return means, log_var2