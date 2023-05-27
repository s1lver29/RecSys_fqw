import torch
from numpy import mean, reciprocal, log2

def hit(y_pred, y_true):
    hr = set(y_pred).intersection({y_true})
    if len(hr) != 0:
        return 1
    return 0


def precision(y_pred, y_true, p=10):
    return len((set(y_pred).intersection({y_true}))) / p


def recall(y_pred, y_true):
    return len(set(y_pred).intersection({y_true})) / len({y_true})


def mrr(y_pred, y_true):
    for i in range(len(y_pred)):
        if y_pred[i] == y_true:
            return 1 / (y_pred.index(y_true) + 1)
    return 0


def map_k(y_pred, y_true):
    relevances = []
    precisions = []

    for i in range(len(y_pred)):
        if y_pred[i] == y_true:
            relevances.append(1)
            precisions.append(sum(relevances) / (i + 1))

    if len(precisions) > 0:
        return mean(precisions)

    return 0


def ndcg(y_pred, y_true):
    if y_true in y_pred:
        index = y_pred.index(y_true)
        return reciprocal(log2(index + 2))
    return 0


@torch.no_grad()
def metrics(model, test_loader, criterion, top_k, device):
    _hr, _precision, _recall, _mrr, _map, _ndcg = [], [], [], [], [], []

    test_loss = []
    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)
        label = label.to(device)

        predictions = model(user, item)
        predictions = predictions.view(-1)
        loss = criterion(predictions.to(torch.float64), label.to(torch.float64))
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()

        y_true = item[0].item()
        _hr.append(hit(recommends, y_true))
        _precision.append(precision(recommends, y_true))
        _recall.append(recall(recommends, y_true))
        _mrr.append(mrr(recommends, y_true))
        _map.append(map_k(recommends, y_true))
        _ndcg.append(ndcg(recommends, y_true))
        test_loss.append(loss.cpu().numpy())

    return (
        mean(test_loss),
        mean(_hr),
        mean(_precision),
        mean(_recall),
        mean(_mrr),
        mean(_map),
        mean(_ndcg),
    )
