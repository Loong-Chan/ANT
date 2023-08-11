import torch
import os
os.environ['DGLBACKEND'] = 'pytorch'
import utils
import numpy as np

def create_raw_dataset(name):

    if name in ["cora", "citeseer", "pubmed"]:
        from torch_geometric.datasets import Planetoid
        Planetoid(root="Dataset/", name=name)
        adj, features, idx_train, idx_val, idx_test, labels = utils.load_data(name)

    elif name in ["chameleon", "squirrel"]:
        import torch_geometric
        from torch_geometric.datasets import WikipediaNetwork
        dataset = WikipediaNetwork(root="data", name=name)
        features = dataset.x
        adj = torch_geometric.utils.to_scipy_sparse_matrix(dataset.edge_index).todense().A
        adj = torch.FloatTensor(adj)
        labels = dataset.y

    elif name in ["computer", "photo"]:
        if name == "computer":
            from dgl.data import AmazonCoBuyComputerDataset
            dataset = AmazonCoBuyComputerDataset()
        elif name == "photo":
            from dgl.data import AmazonCoBuyPhotoDataset
            dataset = AmazonCoBuyPhotoDataset()
        graph = dataset[0]
        features = graph.ndata["feat"]
        labels = graph.ndata["label"]
        adj = graph.adj().to_dense()

    classes = []
    for i in range(labels.max().item()+1):
        classes.append((labels==i).nonzero().squeeze())

    idx_train = torch.cat([c[:20] for c in classes]).to(torch.int64)
    idx_val = torch.cat([c[21:50] for c in classes]).to(torch.int64)
    idx_test = torch.cat([c[51:] for c in classes]).to(torch.int64)

    environment = torch.zeros(size=(adj.shape[0],))
    environment[idx_train] = 1
    environment = torch.nn.functional.one_hot(environment.to(torch.int64)).to(torch.float32)
    dataset = adj, features, labels, idx_train, idx_val, idx_test, environment
    torch.save(dataset, f"Dataset/{name}_1.pth")


def processed_dataset(dataset_name, device):
    filename = f"Dataset/{dataset_name}.pth"
    [adj, features, labels, idx_train, idx_val, idx_test, environment] = torch.load(filename)
    edge_index = (adj + torch.eye(adj.shape[0])).nonzero().T.to(torch.long)
    edge_index = edge_index.to(device)

    features = utils.tensor_rownorm(features)
    adj = adj.numpy() + np.identity(adj.shape[0])
    adj = torch.tensor(utils.numpy_rownorm(adj), dtype=torch.float32)

    edge_index = adj.nonzero().T.to(torch.long)

    adj = adj.to(torch.float32).to(device)
    features = features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    environment = environment.to(device)
    edge_index = edge_index.to(device)

    return adj, features, labels, idx_train, idx_val, idx_test, environment, edge_index


if __name__ == "__main__":
    import sys 
    dataset = sys.argv[1]
    create_raw_dataset(dataset)
