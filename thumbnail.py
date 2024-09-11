import os

import torch

from loader import ThumbnailDataset
from sae_training.utils import ViTSparseAutoencoderSessionloader
from vit_sae_analysis.dashboard_fns import get_feature_data

load_pretrained = True

print('loading data')
parent_dir = 'cruft/cruft/video/'
files = [parent_dir + f for f in os.listdir(parent_dir) if f.endswith('.h5')]
dataset = ThumbnailDataset(files, keys=['thumbnailStandard'], device='cuda')

print('n samples in dataset', len(dataset))

# https://en.wikipedia.org/wiki/List_of_most-subscribed_YouTube_channels
# mrbeast
# tseries
# Cocomelon
# SET India
# Kids Diana Show
# Vlad and Niki
# Like Nastya

# thumbnails @ s3://vit-sae/yt-524_288/
# 30337 - graffiti
# 58040 - jeans 
# 61332 - skateboarding
# 60705 - graffiti
# 33318 - tape recorder
# 3240 - ships
# 56264 - ships
# 44550 - boxing 
# 43231 - tape recorders (nice) 
# 53416 - basketball

# 47588 - graffiti
# 56264 - ships
# 10564 - fish in the process of being fished (RIP)
# 
# 19172 - manly men, 


if load_pretrained:
    get_feature_data(
        None,
        None,
        threshold=0, 
        max_number_of_images_per_iteration=4096,
        number_of_images=524_288,
        number_of_max_activating_images=20,
        dataset=dataset,
        load_pretrained=True,
        # directory='cruft/cifar_dashboard',
        neuron_idx=torch.tensor([60705, 47588, 56264, 10564, 23408, 58040,  3240, 33318, 24765, 47293]),
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
        max_number_of_images_per_iteration=8192,
        number_of_images=524_288,
        number_of_max_activating_images=20,
        dataset=dataset,
        image_key='thumbnailStandard'
    )