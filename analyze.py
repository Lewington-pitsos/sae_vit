import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from loader import ThumbnailDataset
import umap
from matplotlib.colors import LogNorm

for name in [
    # 'cocomelon', 
    # 'kids-diana-show', 
    # 'like-nastya',
    # 'mrbeast',
    'tseries',
    'vlad-and-niki',
    'zee-music-channel'
]:
    if not os.path.exists(f'cruft/{name}'):
        os.makedirs(f'cruft/{name}')

    activation_dir = f'cruft/{name}_dash/raw'
    files = sorted([activation_dir + '/' + f for f in os.listdir(activation_dir)], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for f in files[:10]:
        print(f)

    stack = None
    for f in files:
        data = torch.load(f).T
        if stack is None:
            stack = data
        else:
            stack = torch.cat((stack, data))

    print('activations shape', stack.shape)
    stack = stack.cpu()

    path = os.path.join('channel', name)
    dataset = ThumbnailDataset(path, data_types=['numeric'], device='cpu')

    views = dataset[:]['viewCount']

    print('views shape', views.shape)

    binary_stack = stack * (stack > 0.05)
    binstacksum = binary_stack.sum(dim=0)

    means = torch.zeros(stack.shape[1])

    top_n = 10

    for i in range(stack.shape[1]):
        if binstacksum[i] >= top_n:
            # active_videos = binstack[:, i] > 0
            _, active_videos = torch.topk(stack[:, i], top_n)
            active_vid_views = views[active_videos]

            mean_views = active_vid_views.mean()
            means[i] = mean_views


    print(f'top mean: {means.max():.2e}')
    n_means_gt_zero = (means > 0).sum()
    print('n means greater than zero', n_means_gt_zero)

    random_means = []
    for i in range(n_means_gt_zero):
        rand_idx = torch.randint(0, views.shape[0], (top_n,))

        random_means.append(views[rand_idx].mean())

    print(f'random means: {np.mean(random_means):.2e} +- {np.std(random_means):.2e} --- {np.mean(random_means) + 2 * np.std(random_means):.2e}/{np.mean(random_means) - 2 * np.std(random_means):.2e}')

    max_mean = means.max().item()

    plt.hist(random_means, alpha=0.5, bins=50, range=(1, max_mean))
    plt.hist(means, alpha=0.5, bins=50, range=(1, max_mean))

    plt.savefig(f'cruft/{name}/comp_mean_plot.png')
    plt.close()

    # save topk
    val, idx = torch.topk(means, 10)
    print('topk', val, idx)

    torch.save(val, f'cruft/{name}/topk_val.pt')
    torch.save(idx, f'cruft/{name}/topk_idx.pt')

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(stack.cpu())
    print('embedding shape', embedding.shape)

    plt.figure(figsize=(10, 10))

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=views,  # Use the likes tensor to color the points
        cmap='viridis',
        norm=LogNorm(),
        s=4,
    )
    plt.colorbar()
    plt.gca().set_aspect('equal', 'datalim')

    plt.savefig(f'cruft/{name}/umap2.png')
    plt.close()

