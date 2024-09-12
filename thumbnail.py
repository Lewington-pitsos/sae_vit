import os
import torch

from loader import ThumbnailDataset
from sae_training.utils import ViTSparseAutoencoderSessionloader
from vit_sae_analysis.dashboard_fns import get_feature_data


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

name = 'mrbeast'
print('loading data')
path = os.path.join('channel', name)
dataset = ThumbnailDataset(path, data_types=['thumbnail'], device='cuda')

print('n samples in dataset', len(dataset))

image_key = 'thumbnail'
load_pretrained = True
threshold = 0.03
batch_size=256
directory=f'cruft/{name}_dash'

features = torch.tensor([32979, 25709, 58510, 14073, 30299, 47643, 13881,  8486, 21690, 52368])
if load_pretrained:
    get_feature_data(
        None,
        None,
        threshold=threshold, 
        max_number_of_images_per_iteration=batch_size,
        number_of_images=814,
        number_of_max_activating_images=20,
        directory=directory,
        neuron_idx=features,
        dataset=dataset,
        image_key=image_key,
        load_pretrained=True,
    )
else:
    print('creating new data')
    filename = 'cruft/clip-vit-large-patch14_-2_resid_65536.pt'
    pretrain = torch.load(filename)
    cfg = pretrain['cfg']

    model, sparse_autoencoder = ViTSparseAutoencoderSessionloader.load_essential_from_pretrained(filename)

    get_feature_data(
        sparse_autoencoder,
        model,
        threshold=threshold,
        max_number_of_images_per_iteration=batch_size,
        number_of_images=812,
        number_of_max_activating_images=20,
        directory=directory,
        neuron_idx=features,
        dataset=dataset,
        image_key=image_key
    )