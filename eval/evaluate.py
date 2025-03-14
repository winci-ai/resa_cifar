import torch
import torch.nn as nn
import torch.optim as optim
from .dataloader import get_data

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model
        
def get_acc(model, ds, args):
    model.eval()
    model = unwrap_model(model)
    
    if hasattr(model, 'momentum_encoder'):
        feature = model.momentum_encoder
    else:
        feature = model.encoder
    out_size = model.out_size

    x_train, y_train = get_data(feature, ds.clf, out_size, "cuda")
    x_test, y_test = get_data(feature, ds.test, out_size, "cuda")

    acc_knn = eval_knn(x_train, y_train, x_test, y_test)
    acc_linear = eval_sgd(x_train, y_train, x_test, y_test)

    del x_train, y_train, x_test, y_test
    model.train()
    return acc_knn, acc_linear

def eval_knn(x_train, y_train, x_test, y_test, k=5):
    """ k-nearest neighbors classifier accuracy """
    d = torch.cdist(x_test, x_train)
    topk = torch.topk(d, k=k, dim=1, largest=False)
    labels = y_train[topk.indices]
    pred = torch.empty_like(y_test)
    for i in range(len(labels)):
        x = labels[i].unique(return_counts=True)
        pred[i] = x[0][x[1].argmax()]

    acc = (pred == y_test).float().mean().cpu().item()
    del d, topk, labels, pred
    return acc

def eval_sgd(x_train, y_train, x_test, y_test, topk=[1, 5], epoch=500):
    """ linear classifier accuracy (sgd) """
    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    num_class = y_train.max().item() + 1
    clf = nn.Linear(output_size, num_class)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epoch):
        perm = torch.randperm(len(x_train)).view(-1, 1000)
        for idx in perm:
            optimizer.zero_grad()
            criterion(clf(x_train[idx]), y_train[idx]).backward()
            optimizer.step()
        scheduler.step()

    clf.eval()
    with torch.no_grad():
        y_pred = clf(x_test)
    pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
    acc = {
        t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
        for t in topk
    }
    del clf
    return acc