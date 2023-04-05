import torch
import os
import io
import csv

from MISA.models import MISA
from TAILOR.models import TAILOR
from TAILOR.file_utils import *
from confidNet import ConfidenceRegressionNetwork


def save_load_name(args, name='', confidNet=None, dynamicKT=None):
    if len(name) > 0:
        return name
    
    if args.aligned:
        aligned = name if len(name) > 0 else 'aligned'
    elif not args.aligned:
        aligned = name if len(name) > 0 else 'nonaligned'

    if confidNet==True:
        name = f'confidNet_{args.data}_{aligned}_{args.model}({args.n_epoch}epochs)_{args.conf_loss}_epoch({args.n_epoch_conf})'
        return name

    if dynamicKT==True:
        name = f'model_{args.data}_{aligned}_{args.model}_kt_{args.kt_model}_{args.kt_weight}'
    else:
        name = f'model_{args.data}_{aligned}_{args.model}_baseline_epoch({args.n_epoch})'
        return name


def save_model(args, model, name='', confidNet=None, dynamicKT=None):
    if not os.path.exists('pre_trained_models'):
        os.mkdir('pre_trained_models')
    
    config_name = save_load_name(args, name, confidNet, dynamicKT)
    torch.save(model.state_dict(), f'pre_trained_models/best_{config_name}.pt')


def load_model(args, name='', confidNet=None, dynamicKT=None):
    config_name = save_load_name(args, name, confidNet, dynamicKT)

    with open(f'pre_trained_models/best_{config_name}.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    model_state_dict = torch.load(buffer, map_location='cpu')

    if confidNet==True:
        model = ConfidenceRegressionNetwork(args, input_dims=args.hidden_size*6, num_classes=1, dropout=args.conf_dropout)
        model.load_state_dict(model_state_dict)

    else:
        if args.model == 'MISA':
            model = MISA(args)
        elif args.model == 'TAILOR':
            cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

            model = TAILOR.from_pretrained(args.bert_model, args.visual_model, args.audio_model, args.cross_model,
                                    cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
        
        model.load_state_dict(model_state_dict)

    model = model.to(args.device)

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
