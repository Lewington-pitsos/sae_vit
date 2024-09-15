import os
import torch

from loader import ThumbnailDataset
from sae_training.utils import ViTSparseAutoencoderSessionloader
from vit_sae_analysis.dashboard_fns import get_feature_data
from datasets import load_dataset


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


filename = 'cruft/clip-vit-large-patch14_-2_resid_65536.pt'
pretrain = torch.load(filename)
cfg = pretrain['cfg']

model, sparse_autoencoder = ViTSparseAutoencoderSessionloader.load_essential_from_pretrained(filename)
# model, sparse_autoencoder, _ = ViTSparseAutoencoderSessionloader.load_session_from_pretrained(filename)

for name in [
    # 'cocomelon', 
    # 'kids-diana-show', 
    # 'like-nastya',
    # 'mrbeast',
    # 'tseries',
    # 'vlad-and-niki',
    # 'zee-music-channel'
    'imgnet',
]:  
    if name == 'imgnet':
        dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="train")
        dataset = dataset.shuffle(seed = 1)
        image_key='image'
    else:
        image_key = 'thumbnail'
        print('loading data')
        path = os.path.join('channel', name)
        dataset = ThumbnailDataset(path, data_types=['thumbnail'], device='cuda')

    print('n samples in dataset', len(dataset))

    load_pretrained = True
    threshold = 0.00
    batch_size=256
    directory=f'cruft/{name}_dash'
    max_imgs = 30_000


    # features = torch.tensor([32979, 25709, 58510, 14073, 30299, 47643, 13881,  8486, 21690, 52368]) all mr beast vids
    # features = torch.tensor([45287, 13881, 30299, 50420, 52368,  8486, 13790, 56283, 21690, 12802]) # just mrbeast recent videos
    # features = torch.tensor([61332, 46540, 58255,  9236, 33756, 51259, 11666, 15376, 40998, 19632,
    #     65131, 29049,  8142, 26633, 31448, 46290, 49886, 61094, 42085, 17758,
    #     18248, 30533,  7475, 12212, 15645, 49346, 45160, 13853, 30650, 60706,
    #     29293, 27831, 47293, 43758, 24765, 46794,  3571, 25294, 44983, 31589,
    #     49511,  3029,  5616, 36064, 20134, 61091, 38634, 64295, 23408, 26753])

    # features = torch.load(f'cruft/{name}/topk_idx.pt')

    features = torch.tensor([20293, 59281, 16059,  8400, 35679, 20372,  8971, 14500,  8328, 31451, # mrbeast highest
        
        38303, 25944,  2387,  6415, 26498,  1228, 52191, 36261, 42102, 18158]) # mrbeast lowest
    # features = None
    if load_pretrained:
        get_feature_data(
            None,
            None,
            threshold=threshold, 
            max_number_of_images_per_iteration=batch_size,
            number_of_images=max_imgs,
            number_of_max_activating_images=20,
            directory=directory,
            neuron_idx=features,
            dataset=dataset,
            image_key=image_key,
            load_pretrained=True,
        )
    else:
        print('creating new data')

        get_feature_data(
            sparse_autoencoder,
            model,
            threshold=threshold,
            max_number_of_images_per_iteration=batch_size,
            number_of_images=max_imgs,
            number_of_max_activating_images=20,
            directory=directory,
            neuron_idx=features,
            dataset=dataset,
            image_key=image_key
        )