import time
import torch
from torch import nn
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, optimizer, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)
    before = time.time()
    print("training", len(train_loader), "number of batches")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx == 0:
            first_time = time.time()
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = model(inputs)  # 3D
        loss = criterion(outputs.view(-1, outputs.size(2)), targets.view(-1))  # Loss of the flattened outputs
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print("Time elapsed", time.time() - first_time)

        if batch_idx % 100 == 0 and batch_idx != 0:
            after = time.time()
            print("Time: ", after - before)
            print("Loss per word: ", loss.item() / batch_idx)
            print("Perplexity: ", np.exp(loss.item() / batch_idx))
            after = before

    val_loss = 0
    batch_id = 0
    for inputs, targets in val_loader:
        batch_id += 1
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(2)), targets.view(-1))
        val_loss += loss.item()
    val_lpw = val_loss / batch_id
    print("\nValidation loss per word:", val_lpw)
    print("Validation perplexity :", np.exp(val_lpw), "\n")
    return val_lpw


def train_epoch_packed(model, optimizer, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(
        reduction="sum")  # sum instead of averaging, to take into account the different lengths
    criterion = criterion.to(DEVICE)
    batch_id = 0
    before = time.time()
    print("Training", len(train_loader), "number of batches")
    for inputs, targets in train_loader:  # lists, presorted, preloaded on GPU
        batch_id += 1
        outputs = model(inputs)
        loss = criterion(outputs, torch.cat(targets))  # criterion of the concatenated output
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
            after = time.time()
            nwords = np.sum(np.array([len(l) for l in inputs]))
            lpw = loss.item() / nwords
            print("Time elapsed: ", after - before)
            print("At batch", batch_id)
            print("Training loss per word:", lpw)
            print("Training perplexity :", np.exp(lpw))
            before = after

    val_loss = 0
    batch_id = 0
    nwords = 0
    for inputs, targets in val_loader:
        nwords += np.sum(np.array([len(l) for l in inputs]))
        batch_id += 1
        outputs = model(inputs)
        loss = criterion(outputs, torch.cat(targets))
        val_loss += loss.item()
    val_lpw = val_loss / nwords
    print("\nValidation loss per word:", val_lpw)
    print("Validation perplexity :", np.exp(val_lpw), "\n")
    return val_lpw
