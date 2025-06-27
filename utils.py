import numpy as np
from sklearn.metrics import (auc, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_curve, roc_auc_score)
import torch as th
import torch.nn.functional as F
def get_metric(y_true, y_pred, y_prob):

    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    Auc = auc(fpr, tpr)

    precision1, recall1, _ = precision_recall_curve(y_true, y_prob)
    Aupr = auc(recall1, precision1)

    return Auc, Aupr, accuracy, precision, recall, f1, mcc

def projection(args, z: th.Tensor) -> th.Tensor:
    fc1 = th.nn.Linear(args.num_hidden, args.num_proj_hidden1).to(args.device)
    fc2 = th.nn.Linear(args.num_proj_hidden1, args.num_proj_hidden2).to(args.device)
    fc3 = th.nn.Linear(args.num_proj_hidden2, args.num_hidden).to(args.device)
    z1 = F.elu(fc1(z))
    z2 = F.elu(fc2(z1))
    # z = th.sigmoid(fc1(z))
    return fc3(z2)
def sim(z1: th.Tensor, z2: th.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return th.mm(z1, z2.t())

def semi_loss(args, z1: th.Tensor, z2: th.Tensor, flag: int):
    # if flag == 0:
    #     f = lambda x: th.exp(x / args.tau_drug)
    # else:
    #     f = lambda x: th.exp(x / args.tau_disease)
    f = lambda x: th.exp(x / args.tau)
    refl_sim = f(args.intra * sim(z1, z1))  # torch.Size([663, 663])
    between_sim = f(args.inter * sim(z1, z2))  # z1 z2:torch.Size([663, 75])
    # refl_sim = f(sim(z1, z1))  # torch.Size([663, 663])
    # between_sim = f(sim(z1, z2))  # z1 z2:torch.Size([663, 75])
    # refl_sim = (F.cosine_similarity(z1, z1))  # torch.Size([663])
    # between_sim = f(F.cosine_similarity(z1, z2))

    return -th.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def batched_semi_loss(args, z1: th.Tensor, z2: th.Tensor,
                        batch_size: int):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: th.exp(x / args.tau)
    indices = th.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = f(sim(z1[mask], z1))  # [B, N]
        between_sim = f(sim(z1[mask], z2))  # [B, N]

        losses.append(-th.log(
            between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            / (refl_sim.sum(1) + between_sim.sum(1)
                - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

    return th.cat(losses)


def LOSS(args, z1: th.Tensor, z2: th.Tensor,
        mean: bool = True, batch_size: int = 0, flag: int = 0):
    h1 = projection(args, z1)
    h2 = projection(args, z2)

    if batch_size == 0:
        l1 = semi_loss(args, h1, h2, flag)
        l2 = semi_loss(args, h2, h1, flag)
    else:
        l1 = batched_semi_loss(h1, h2, batch_size)
        l2 = batched_semi_loss(h2, h1, batch_size)
    # if batch_size == 0:
    #     l1 = semi_loss(args, z1, z2)
    #     l2 = semi_loss(args, z2, z1)
    # else:
    #     l1 = batched_semi_loss(args, z1, z2, batch_size)
    #     l2 = batched_semi_loss(args, z2, z1, batch_size)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret