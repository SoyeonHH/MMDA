from torch.autograd import Function, Variable
import torch.nn as nn
import torch
import numpy as np

from utils import to_gpu, to_cpu
from utils.tools import *
from utils.eval import *

"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1 = torch.nan_to_num(input1)
        input2 = torch.nan_to_num(input2)
        
        input1_mean = torch.mean(input1, dim=0, keepdim=True)
        input2_mean = torch.mean(input2, dim=0, keepdim=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


def getBinaryTensor(imgTensor, boundary = 0.35):
    one = torch.ones_like(imgTensor)
    zero = torch.zeros_like(imgTensor)
    return torch.where(imgTensor > boundary, one, zero)


def get_cls_loss(config, predicted_scores, emo_label):
    criterion = nn.BCELoss(reduction="mean")

    if config.data == "ur_funny":
        emo_label = emo_label.squeeze()
    
    emo_label = emo_label.type(torch.float)

    predicted_scores, emo_label = torch.permute(predicted_scores, (1, 0)), torch.permute(emo_label, (1, 0)) # (num_classes, batch_size)

    cls_loss = 0.0

    # summation of loss for each label
    for i in range(emo_label.size(0)):
        cls_loss += criterion(predicted_scores[i], emo_label[i])

    return cls_loss


def get_domain_loss(config, domain_pred_t, domain_pred_v, domain_pred_a):

    criterion = nn.CrossEntropyLoss(reduction="mean")

    if config.use_cmd_sim:
        return 0.0

    # True domain labels
    domain_true_t = to_gpu(torch.LongTensor([0]*domain_pred_t.size(0)))
    domain_true_v = to_gpu(torch.LongTensor([1]*domain_pred_v.size(0)))
    domain_true_a = to_gpu(torch.LongTensor([2]*domain_pred_a.size(0)))

    # Stack up predictions and true labels
    domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
    domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

    return criterion(domain_pred, domain_true)

def get_cmd_loss(config, utt_shared_t, utt_shared_v, utt_shared_a):

    loss_cmd = CMD()

    if not config.use_cmd_sim:
        return 0.0

    # losses between shared states
    loss = loss_cmd(utt_shared_t, utt_shared_v, 5)
    loss += loss_cmd(utt_shared_t, utt_shared_a, 5)
    loss += loss_cmd(utt_shared_a, utt_shared_v, 5)
    loss = loss/3.0

    return loss

def get_diff_loss(utt_shared, utt_private):

    loss_diff = DiffLoss()

    shared_t = utt_shared[0]
    shared_v = utt_shared[1]
    shared_a = utt_shared[2]
    private_t = utt_private[0]
    private_v = utt_private[1]
    private_a = utt_private[2]

    # Between private and shared
    loss = loss_diff(private_t, shared_t)
    loss += loss_diff(private_v, shared_v)
    loss += loss_diff(private_a, shared_a)

    # Across privates
    loss += loss_diff(private_a, private_t)
    loss += loss_diff(private_a, private_v)
    loss += loss_diff(private_t, private_v)

    return loss

def get_recon_loss(utt_recon, utt_orig):

    # self.loss_recon = MSE()
    loss_recon = nn.MSELoss(reduction="mean")

    loss = loss_recon(utt_recon[0], utt_orig[0])
    loss += loss_recon(utt_recon[1], utt_orig[1])
    loss += loss_recon(utt_recon[2], utt_orig[2])
    loss = loss/3.0
    return loss


def get_conf_loss(config, pred, truth, predicted_tcp):    # pred: (batch_size, num_classes), truth: (batch_size, num_classes)
    
    loss_mcp = nn.CrossEntropyLoss(reduction="mean")
    loss_tcp = nn.MSELoss(reduction="mean")
    
    tcp_loss = 0.0
    mcp_loss = 0.0
    tcp_batch = []

    for i in range(truth.size(0)):  # for each batch
        tcp = 0.0
        for j in range(truth[i].size(0)):   # for each class
            tcp += pred[i][j] * truth[i][j]
        tcp = tcp / torch.count_nonzero(truth[i]) if torch.count_nonzero(truth[i]) != 0 else 0.0
        tcp_batch.append(tcp)
    
    tcp_batch = to_gpu(torch.tensor(tcp_batch))
    tcp_loss = loss_tcp(predicted_tcp, tcp_batch)

    # pred, truth = torch.permute(pred, (1, 0)), torch.permute(truth, (1, 0)) # (num_classes, batch_size)

    mcp_loss = loss_mcp(pred, truth)

    # for i in range(truth.size(0)):
    #     mcp_loss += self.loss_mcp(pred[i], truth[i])
    # mcp_loss = mcp_loss / truth.size(0)
    
    if config.use_mcp:
        return torch.add(tcp_loss, mcp_loss, alpha=config.mcp_weight)
    else:
        return tcp_loss


def cosine_similarity_loss(source_net, target_net, dim=1, eps=1e-8):
    
    # Normalize each vector by its norm
    source_net_norm = torch.sqrt(torch.sum(source_net**2, dim=dim, keepdim=True))
    source_net = source_net / (source_net_norm + eps)
    source_net[source_net != source_net] = 0    # replace nan with 0

    target_net_norm = torch.sqrt(torch.sum(target_net**2, dim=dim, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate cosine similarity
    source_similarity = torch.mm(source_net, source_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to [0, 1]
    source_similarity = (source_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    source_similarity = source_similarity / torch.sum(source_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate KL divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (source_similarity + eps)))

    return loss


def supervised_loss(source_net, targets, eps=1e-8):
    
    labels = targets.cpu().numpy()
    target_sim = np.zeros((labels.shape[0], labels.shape[0]), dtype='float32')
    for i in range(labels.shape[0]):
        for j in range(labels.shape[0]):
            target_sim[i, j] = 1.0 if labels[i] == labels[j] else 0.0
    
    target_similarity = torch.from_numpy(target_sim).cuda()
    target_similarity = Variable(target_similarity)

    # Normalize each vector by its norm
    source_net_norm = torch.sqrt(torch.sum(source_net**2, dim=1, keepdim=True))
    source_net = source_net / (source_net_norm + eps)
    source_net[source_net != source_net] = 0    # replace nan with 0

    # Calculate cosine similarity
    source_similarity = torch.mm(source_net, source_net.transpose(0, 1))

    # Scale cosine similarity to [0, 1]
    source_similarity = (source_similarity + 1.0) / 2.0

    # Transform them into probabilities
    source_similarity = source_similarity / torch.sum(source_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate KL divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (source_similarity + eps)))

    return loss