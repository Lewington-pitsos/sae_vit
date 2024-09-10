import torch
from loader import ThumbnailDataset
import os
import time

parent_dir = 'cruft/cruft/video/'
files = [parent_dir + f for f in os.listdir(parent_dir) if f.endswith('.h5')]

device = 'cpu'

dataset = ThumbnailDataset(files, keys=None, device=device)

# sync to gpu time

torch.cuda.synchronize()
start = time.time()
for idx in range(10):
    samples = dataset._get_whole_file(idx, keys=['thumbnailStandard'])

torch.cuda.synchronize()
end = time.time()

print('Time taken:', end - start)

torch.cuda.synchronize()
start = time.time()

for idx in range(10):
    x = torch.load('cruft/compact/' + str(idx) + '.pt', map_location=device)
torch.cuda.synchronize()
end = time.time()
print('Time taken:', end - start)