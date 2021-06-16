import copy

import torch

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        # Deepcopy as I don't want to remove the overflow from the real encodings
        self.encodings = copy.deepcopy(encodings)
        self.encodings.pop('overflow_to_sample_mapping')
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        else:
            item['labels'] = None
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])