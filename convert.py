import torch
from loader import ThumbnailDataset
import os

parent_dir = 'cruft/cruft/video/'
files = [parent_dir + f for f in os.listdir(parent_dir) if f.endswith('.h5')]

dataset = ThumbnailDataset(files, keys=None, device='cpu')

for idx in range(10):
    samples = dataset._get_whole_file(idx, keys=['thumbnailStandard'])

    # data_tensor = torch.stack([
    #     samples['likeCount'], 
    #     samples['viewCount'], 
    #     samples['commentCount']
    # ])

    torch.save(samples['thumbnailStandard'], f'cruft/compact/{idx}.pt')

