from torch.autograd import Function, Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def getBinaryTensor(imgTensor, boundary = 0.35):
    one = torch.ones_like(imgTensor)
    zero = torch.zeros_like(imgTensor)
    return torch.where(imgTensor > boundary, one, zero)


def get_cls_loss(predicted_scores, emo_label):
    criterion = nn.BCELoss(reduction="mean")
    
    emo_label = emo_label.type(torch.float)

    predicted_scores, emo_label = torch.permute(predicted_scores, (1, 0)), torch.permute(emo_label, (1, 0)) # (num_classes, batch_size)

    cls_loss = 0.0

    # summation of loss for each label
    for i in range(emo_label.size(0)):
        cls_loss += criterion(predicted_scores[i], emo_label[i])

    return cls_loss


def get_kt_loss(t, v, a, label, dynamic_weight=None, supervised_weights=0):
    '''
    shape of t: (batch_size, hidden_size)
    shape of v: (batch_size, hidden_size)
    shape of a: (batch_size, hidden_size)
    shape of label: (batch_size, num_classes=6)
    '''

    if dynamic_weight is None:
        dynamic_weight = [1, 1, 1, 1, 1, 1]
    
    loss_t_v = torch.mean(dynamic_weight[0] * cosine_similarity_loss(t, v))
    loss_t_a = torch.mean(dynamic_weight[1] * cosine_similarity_loss(t, a))
    
    loss_v_t = torch.mean(dynamic_weight[2] * cosine_similarity_loss(v, t))
    loss_v_a = torch.mean(dynamic_weight[3] * cosine_similarity_loss(v, a))

    loss_a_t = torch.mean(dynamic_weight[4] * cosine_similarity_loss(a, t))
    loss_a_v = torch.mean(dynamic_weight[5] * cosine_similarity_loss(a, v))

    kt_loss = loss_t_v + loss_t_a + loss_v_t + loss_v_a + loss_a_t + loss_a_v

    return kt_loss.squeeze()


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


def distillation_loss(output, target, T):
    """
    Distillation Loss
    :param output:
    :param target:
    :param T:
    :return:
    """
    output = F.log_softmax(output / T)
    target = F.softmax(target / T)
    loss = -torch.sum(target * output) / output.size()[0]
    return loss
