import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import jaccard_score

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

sys.path.append('./')
sys.path.append('../')
sys.path.append('../src')

from models import VAE
from datasets import InteractionDataset

import seaborn as sns

DATA_DIR = '/home/gridsan/lguan/zhang/PCSM-VAE/data'
MODEL_DIR = '/home/gridsan/lguan/zhang/PCSM-VAE/models'

def evaluate(test_path, model_path, use_cuda=False, jaccard=False):
    """Returns a dictionary of:
    
    x: original data
    recon: sampled reconstruction after passing through VAE
    label: whatever the target label was in the original data (used when you want to
           see whether the VAE could cluster between targets, like protein in the condensate,
           etc.)
    z: latent representation of each input.

    """
    pyro.clear_param_store()
    model_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    z_dim = model_checkpoint['z_dim']
    input_dim = model_checkpoint['input_dim']
    hidden_dim = model_checkpoint['hidden_dim']

    vae = VAE(input_dim, z_dim=z_dim, hidden_dim=hidden_dim, use_cuda=use_cuda)
    vae.load_state_dict(model_checkpoint['state_dict'])
    vae.eval()
    
    test_ds = InteractionDataset(file_path=test_path)
    len_ds = len(test_ds)
    test_dataloader = DataLoader(test_ds,
                                 batch_size=len_ds
                                )
    scores = np.empty([len_ds, len_ds])
    orig_scores = np.empty([len_ds, len_ds])
    with torch.no_grad():
        for x, label in test_dataloader:
            z = vae.encode(x, with_sampling=False).numpy()
            recon = vae.reconstruct(x, with_sampling=False).numpy()
            thresholded_recon = [apply_threshold(p) for p in recon]
            if jaccard:
                for i in tqdm(range(len(x))):
                    for j in range(len(x)):
                        score = jaccard_score(x[i].numpy(), thresholded_recon[j])
                        orig_score = jaccard_score(x[i].numpy(), x[j])
                        scores[i][j] = score
                        orig_scores[i][j] = orig_score
                return {'x': x.numpy(),
                        'recon': recon,
                        'label': label,
                        'scores': scores,
                        'binary_pred': thresholded_recon,
                        'original_scores': orig_scores,
                        'z': z
                       }
            else:
                return {'x': x.numpy(),
                        'recon': recon,
                        'label': label,
                        'z': z
                       }

        
def apply_threshold(arr, threshold=0.15):
    """Binarizes `arr` using the specified threshold.
    
    Args:
        arr: float or iterable of values between 0 and 1
        threshold: float between 0 and 1
    """
    try:
        arr_ = iter(arr)
    except TypeError:
        if arr >= 0 and arr <= 1:
            return 1 if arr >= threshold else 0
        else:
            raise ValueError('Input(s) must be between 0 and 1.')

    output = []
    for i in range(len(arr)):
        if arr[i] >= 0 and arr[i] <= 1:
            val = 1 if arr[i] >= threshold else 0
            output.append(val)
        else:
            raise ValueError('Input(s) must be between 0 and 1.')
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCSM VAE encoding parameters')

    parser.add_argument('model',
                        type=str, 
                        help='name of file containing model')
    parser.add_argument('val_data',
                        type=str, 
                        help='name of validation data file')

    parser.add_argument('--use_cuda',
                        default=False,
                        action='store_true',
                        help='whether to use cuda for GPU'
                       )


    args = vars(parser.parse_args())

    output_dict = evaluate(test_path=os.path.join(DATA_DIR, args['val_data']), 
                           model_path=os.path.join(MODEL_DIR, args['model']), 
                           jaccard=False)

    df = pd.DataFrame(output_dict['label'].numpy())
    df['recon'] = pd.Series(output_dict['recon'].tolist())
    df['z'] = pd.Series(output_dict['z'].tolist())

    output_name = f'vae_output_{args["val_data"]}.pkl'
    df.to_pickle(os.path.join(DATA_DIR, output_name))
