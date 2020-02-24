import numpy as np
import torch

import machine_learning.pytorch.rnn.shakespeare_data as sh
# Data - refer to shakespeare_data.py for details
from machine_learning.pytorch.rnn.models import PackedLanguageModel
from machine_learning.pytorch.rnn.textdataset import LinesDataset, collate_lines, get_dataloader
from machine_learning.pytorch.rnn.training import DEVICE, train_epoch_packed


def generate(model, seed, nwords):
    seq = sh.map_corpus(seed, charmap)
    seq = torch.tensor(seq).to(DEVICE)
    out = model.generate(seq, nwords)
    return sh.to_text(out.cpu().detach().numpy(), chars)


# read corpus
corpus = sh.read_corpus()
chars, charmap = sh.get_charmap(corpus)
charcount = len(chars)
shakespeare_array = sh.map_corpus(corpus, charmap)
small_example = shakespeare_array[:17]

# init model and dataset
# model = CharLanguageModel(charcount, 256, 256, 3)
# model = model.to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
# split = 5000000
# train_dataset = TextDataset(shakespeare_array[:split])
# val_dataset = TextDataset(shakespeare_array[split:])
# train_loader = get_dataloader(train_dataset)
# val_loader = get_dataloader(val_dataset)
# for i in range(1):
#     train_epoch(model, optimizer, train_loader, val_loader)
#     print(generate(model, "Ayou", 3))

stop_character = charmap['\n']
space_character = charmap[" "]
lines = np.split(shakespeare_array, np.where(shakespeare_array == stop_character)[0] + 1)  # split the data in lines
shakespeare_lines = []
for s in lines:
    s_trimmed = np.trim_zeros(s - space_character) + space_character  # remove space-only lines
    if len(s_trimmed) > 1:
        shakespeare_lines.append(s)
for i in range(10):
    print(sh.to_text(shakespeare_lines[i], chars))
print(len(shakespeare_lines))
model = PackedLanguageModel(charcount, 256, 256, 3, stop=stop_character)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
split = 100000
train_dataset = LinesDataset(shakespeare_lines[:split])
val_dataset = LinesDataset(shakespeare_lines[split:])
train_loader = get_dataloader(train_dataset, collate_lines)
val_loader = get_dataloader(val_dataset, collate_lines)
print(generate(model, "To be, or not to be, that is the q", 20))
for i in range(2):
    train_epoch_packed(model, optimizer, train_loader, val_loader)
torch.save(model, "trained_model.pt")
