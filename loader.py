import h5py
from torch.utils.data import Dataset
import os
import torch

class ThumbnailDataset(Dataset):
    def __init__(self, file_paths, device, keys=None):
        self.file_paths = file_paths  # List of HDF5 file paths
        self.index_map = []
        self.start_end = {}
        for file_idx, file_path in enumerate(self.file_paths):
            initial_idx = len(self.index_map)
            with h5py.File(file_path, 'r') as hf:
                keys = list(hf.keys())
                # Assuming all datasets have the same length for simplicity
                num_samples = len(hf[keys[0]])
                for i in range(num_samples):
                    self.index_map.append((file_idx, i))

            self.start_end[file_idx] = (initial_idx, len(self.index_map))
                
        self.device = device
        self.keys = keys
        

    def all_file_indices(self):
        return list(range(len(self.file_paths)))

    def __len__(self):
        return len(self.index_map)

    def _get_item(self, idx, keys):
        return self._get_file_slice(idx, idx + 1, keys)

    def _get_file_slice(self, absolute_start, absolute_end, wanted_keys=None):
        file_idx, _ = self.index_map[absolute_start]
        file_start, _ = self.start_end[file_idx]


        relative_start = absolute_start - file_start
        relative_end = absolute_end - file_start
        file_path = self.file_paths[file_idx]

        with h5py.File(file_path, 'r') as hf:
            keys = list(hf.keys())
            if wanted_keys is not None:
                keys = [key for key in keys if key in wanted_keys]
            data = {
                'thumbnailStandard': torch.tensor(hf['thumbnailStandard'][relative_start:relative_end], device=self.device),
                'likeCount': torch.tensor(hf['likeCount'][relative_start:relative_end], device=self.device) if 'likeCount' in keys else None,
                'viewCount': torch.tensor(hf['viewCount'][relative_start:relative_end], device=self.device) if 'viewCount' in keys else None,
                'commentCount': torch.tensor(hf['commentCount'][relative_start:relative_end], device=self.device) if 'commentCount' in keys else None,
                'title': list(hf['title'][relative_start:relative_end]) if 'title' in keys else None
            }
        return data

    def _get_whole_file(self, file_idx, keys):
        start, end = self.start_end[file_idx]
        return self._get_file_slice(start, end, keys)

    def _return_multi_file_slice(self, start, end, keys):
        all_indices = self.index_map[start:end]

        whole_files = list()

        last_file_idx = None
        first_file_idx = None
        for file_idx, _ in all_indices:
            if file_idx not in whole_files:
                whole_files.append(file_idx)
            last_file_idx = file_idx
            
            if first_file_idx is None:
                first_file_idx = file_idx

        if first_file_idx == last_file_idx:
            return self._get_file_slice(start, end, keys)

        whole_files = [idx for idx in whole_files if idx != first_file_idx and idx != last_file_idx]

        sample_lists = []
    
        _, end_of_first_file = self.start_end[first_file_idx]
        first_file_samples = self._get_file_slice(start, end_of_first_file, keys)
        sample_lists.append(first_file_samples)

        for file_idx in whole_files:
            sample_lists.append(self._get_whole_file(file_idx, keys))

        start_of_last_file, _ = self.start_end[last_file_idx]
        last_file_samples = self._get_file_slice(start_of_last_file, end, keys)
        sample_lists.append(last_file_samples)
        
        data = None
        for samples in sample_lists:
            if data is None:
                data = samples
            else:
                for key in data.keys():
                    if key == 'title':
                        data[key] = data[key] + samples[key]
                    else:    
                        data[key] = torch.cat((data[key], samples[key]), 0)

        return data


    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.index_map)

            if idx.step is not None:
                raise ValueError("Slicing with step is not supported", idx)

            return self._return_multi_file_slice(start, stop, self.keys)

        return self._get_item(idx, self.keys)