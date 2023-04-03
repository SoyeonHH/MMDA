import torch
import os
import io
import csv


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name='', confidNet=None, dynamicKT=None):
    if not os.path.exists('pre_trained_models'):
        os.mkdir('pre_trained_models')
    
    if confidNet is not None:
        torch.save(model.state_dict(), f'pre_trained_models/best_confidNet_{args.data}_{name}({args.n_epoch}epochs)_{args.conf_loss}_epoch({args.n_epoch_conf}).pt')
        return

    if dynamicKT is not None:
        torch.save(model.state_dict(), f'pre_trained_models/best_model_{args.data}_{name}_kt_{args.kt_model}_{args.kt_weight}.pt')
    else:
        torch.save(model.state_dict(), f'pre_trained_models/best_model_{args.data}_{name}_baseline_epoch({args.n_epoch}).pt')


def load_model(args, name='', confidNet=None, dynamic_KT=None):
    if confidNet is not None:
        file = f'pre_trained_models/best_confidNet_{args.data}_{name}({args.n_epoch}epochs)_{args.conf_loss}_epoch({args.n_epoch_conf}).pt'
        with open(file, 'rb') as f:
            buffer = io.BytesIO(f.read())
        model = torch.load(buffer)
        return model

    if dynamic_KT is not None:
        file = f'pre_trained_models/best_model_{args.data}_{name}_kt_{args.kt_model}_{args.kt_weight}.pt'
    else:
        file = f'pre_trained_models/best_model_{args.data}_{name}_baseline_epoch({args.n_epoch}).pt'
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

def save_results(args, results, mode=None):
    if not os.path.exists('results'):
        os.mkdir('results')
    
    file_name = f'/results/{args.data}_{args.model}_confidNet_{args.conf_loss}_epoch({args.n_epoch_conf})_{mode}_results_process_all_zero_version.csv'
    
    with open(os.getcwd() + file_name, mode='w') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        for d in results:
            writer.writerow(d)
