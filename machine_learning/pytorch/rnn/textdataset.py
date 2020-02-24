# Dataset class. Transform raw text into a set of sequences of fixed length, and extracts inputs and targets
import torch
from torch.utils.data import Dataset, DataLoader

from machine_learning.pytorch.rnn.training import DEVICE


def get_dataloader(dataset, collate_fn):
    return DataLoader(dataset, shuffle=True, batch_size=64, collate_fn=collate_fn)


# Fixed line dataset
class TextDataset(Dataset):
    def __init__(self, text, seq_len=200):
        n_seq = len(text) // seq_len
        text = text[:n_seq * seq_len]
        self.data = torch.tensor(text).view(-1, seq_len)

    def __getitem__(self, i):
        txt = self.data[i]
        return txt[:-1], txt[1:]

    def __len__(self):
        return self.data.size(0)


def collate(seq_list):
    inputs = torch.cat([s[0].unsqueeze(1) for s in seq_list], dim=1)
    targets = torch.cat([s[1].unsqueeze(1) for s in seq_list], dim=1)
    return inputs, targets


class LinesDataset(Dataset):
    def __init__(self, lines):
        self.lines = [torch.tensor(l) for l in lines]

    def __getitem__(self, i):
        line = self.lines[i]
        return line[:-1].to(DEVICE), line[1:].to(DEVICE)

    def __len__(self):
        return len(self.lines)


# collate fn lets you control the return value of each batch
# for packed_seqs, you want to return your data sorted by length
def collate_lines(seq_list):
    inputs, targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs, targets
