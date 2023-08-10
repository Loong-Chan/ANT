import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer):
        super(MLP, self).__init__()
        layer_size = [nfeat] + [nhid] * (nlayer - 1) + [nclass]
        self.layers = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_size[:-1], layer_size[1:])):
            self.layers.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i < nlayer - 1:
                self.layers.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, x, *args, **kwargs):
        x = self.layers(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    a = MLP(5,3,2,3)
    for p in a.parameters():
        print(p)
