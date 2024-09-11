import torch
from loader import ThumbnailDataset
import os

parent_dir = 'cruft/cruft/video/'
files = [parent_dir + f for f in os.listdir(parent_dir) if f.endswith('.h5')]

dataset = ThumbnailDataset(files, keys=None, device='cpu')

big_tensor = None
for idx in range(40):
    samples = dataset._get_whole_file(idx, keys=['likeCount', 'viewCount', 'commentCount'])

    data_tensor = torch.stack([
        samples['viewCount'], 
        samples['likeCount'], 
        samples['commentCount']
    ], dim=1)

    print(data_tensor.shape)

    # torch.save(data_tensor, f'cruft/compact/stats_{idx}.pt')

    # torch.save(samples['thumbnailStandard'], f'cruft/compact/{idx}.pt')
    
    if big_tensor is None:
        big_tensor = data_tensor
    else:
        big_tensor = torch.concat([big_tensor, data_tensor], dim=0)
    print(big_tensor.shape)


torch.save(big_tensor, 'cruft/compact/big_tensor.pt')
