import os
import torch

from vit_sae_analysis.dashboard_fns import get_feature_data
from sae_training.utils import ViTSparseAutoencoderSessionloader
from loader import ThumbnailDataset

load_pretrained = True

print('loading data')
parent_dir = 'cruft/cruft/video/'
files = [parent_dir + f for f in os.listdir(parent_dir) if f.endswith('.h5')]
dataset = ThumbnailDataset(files, keys=['thumbnailStandard'], device='cuda')

print('n samples in dataset', len(dataset))

if load_pretrained:
    get_feature_data(
        None,
        None,
        threshold=0.04,
        max_number_of_images_per_iteration=4096,
        number_of_images=524_288,
        number_of_max_activating_images=20,
        dataset=dataset,
        load_pretrained=True,
        image_key='thumbnailStandard'
    )
else:
    print('creating new data')
    filename = 'checkpoints/4rfb746w/final_sparse_autoencoder_openai/clip-vit-large-patch14_-2_resid_65536.pt'
    pretrain = torch.load(filename)
    cfg = pretrain['cfg']

    model, sparse_autoencoder, activations_loader = ViTSparseAutoencoderSessionloader.load_session_from_pretrained(filename)

    get_feature_data(
        sparse_autoencoder,
        model,
        threshold=0.04,
        max_number_of_images_per_iteration=4096,
        number_of_images=524_288,
        number_of_max_activating_images=20,
        dataset=dataset,
        image_key='thumbnailStandard'
    )