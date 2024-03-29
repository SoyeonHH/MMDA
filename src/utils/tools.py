import torch
import os
import io


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, dataset=''):
    if not os.path.exists('pre_trained_models'):
        os.mkdir('pre_trained_models')
    if args.use_confidNet:
        torch.save(model.state_dict(), f'pre_trained_models/best_model_MISA_C_{dataset}.pt')
    else:
        torch.save(model.state_dict(), f'pre_trained_models/best_model_MISA_{dataset}.pt')


def load_model(args, dataset=''):
    if args.use_confidNet:
        file = f'pre_trained_models/best_model_MISA_C_{dataset}.pt'
    else:
        file = f'pre_trained_models/best_model_MISA_{dataset}.pt'

    with open(file, 'rb') as f:
        buffer = io.BytesIO(f.read())
    model = torch.load(buffer)

    return model


def random_shuffle(tensor, dim=0):
    if dim != 0:
        perm = (i for i in range(len(tensor.size())))
        perm[0] = dim
        perm[dim] = 0
        tensor = tensor.permute(perm)
    
    idx = torch.randperm(t.size(0))
    t = tensor[idx]

    if dim != 0:
        t = t.permute(perm)
    
    return t

def save_hidden(args, tensor, dataset=''):
    if args.use_confidNet:
        file = f'hidden_vectors/MISA_C_{dataset}.pt'
    else:
        file = f'hidden_vectors/MISA_{dataset}.pt'

    if not os.path.exists('hidden_vectors'):
        os.mkdir('hidden_vectors')
    torch.save(tensor, f'hidden_vectors/MISA_{dataset}.pt')


def load_hidden(args, dataset=''):
    if args.use_confidNet:
        file = f'hidden_vectors/MISA_C_{dataset}.pt'
    else:
        file = f'hidden_vectors/MISA_{dataset}.pt'

    with open(file, 'rb') as f:
        buffer = io.BytesIO(f.read())
    H = torch.load(buffer)
    return H

# TODO: Implement tcp saving
# def save_tcp()