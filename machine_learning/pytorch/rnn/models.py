import torch
from torch import nn


# model for predicting the next character
# process fixed length sequences
from torch.nn.utils import rnn


class CharLanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, nlayers):
        super(CharLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=nlayers)  # Recurrent network
        self.scoring = nn.Linear(hidden_size, vocab_size)  # Projection layer

    def forward(self, seq_batch):  # L x N
        # returns 3D logits
        batch_size = seq_batch.size(1)
        embed = self.embedding(seq_batch)  # L x N x E
        hidden = None
        output_lstm, hidden = self.rnn(embed, hidden)  # L x N x H
        output_lstm_flatten = output_lstm.view(-1, self.hidden_size)  # (L*N) x H
        output_flatten = self.scoring(output_lstm_flatten)  # (L*N) x V
        return output_flatten.view(-1, batch_size, self.vocab_size)

    def generate(self, seq, n_words):  # L x V
        # performs greedy search to extract and return words (one sequence).
        generated_words = []
        embed = self.embedding(seq).unsqueeze(1)  # L x 1 x E
        hidden = None
        output_lstm, hidden = self.rnn(embed, hidden)  # L x 1 x H
        output = output_lstm[-1]  # 1 x H
        scores = self.scoring(output)  # 1 x V
        _, current_word = torch.max(scores, dim=1)  # 1 x 1
        generated_words.append(current_word)
        if n_words > 1:
            for i in range(n_words - 1):
                embed = self.embedding(current_word).unsqueeze(0)  # 1 x 1 x E
                output_lstm, hidden = self.rnn(embed, hidden)  # 1 x 1 x H
                output = output_lstm[0]  # 1 x H
                scores = self.scoring(output)  # V
                _, current_word = torch.max(scores, dim=1)  # 1
                generated_words.append(current_word)
        return torch.cat(generated_words, dim=0)


# Model that takes packed sequences in training
# Process variable length sequences
class PackedLanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, nlayers, stop):
        super(PackedLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                           num_layers=nlayers)  # 1 layer, batch_size = False
        self.scoring = nn.Linear(hidden_size, vocab_size)
        self.stop = stop  # stop line character (\n)

    def forward(self, seq_list):  # list
        batch_size = len(seq_list)
        lens = [len(s) for s in seq_list]  # lens of all lines (already sorted)
        bounds = [0]
        for l in lens:
            bounds.append(bounds[-1] + l)  # bounds of all lines in the concatenated sequence
        seq_concat = torch.cat(seq_list)  # concatenated sequence
        embed_concat = self.embedding(seq_concat)  # concatenated embeddings
        embed_list = [embed_concat[bounds[i]:bounds[i + 1]] for i in range(batch_size)]  # embeddings per line
        packed_input = rnn.pack_sequence(embed_list)  # packed version
        hidden = None
        output_packed, hidden = self.rnn(packed_input, hidden)
        output_padded, _ = rnn.pad_packed_sequence(output_packed)  # unpacked output (padded)
        output_flatten = torch.cat([output_padded[:lens[i], i] for i in range(batch_size)])  # concatenated output
        scores_flatten = self.scoring(output_flatten)  # concatenated logits
        return scores_flatten  # return concatenated logits

    def generate(self, seq, n_words):  # L x V
        generated_words = []
        embed = self.embedding(seq).unsqueeze(1)  # L x 1 x E
        hidden = None
        output_lstm, hidden = self.rnn(embed, hidden)  # L x 1 x H
        output = output_lstm[-1]  # 1 x H
        scores = self.scoring(output)  # 1 x V
        _, current_word = torch.max(scores, dim=1)  # 1 x 1
        generated_words.append(current_word)
        if n_words > 1:
            for i in range(n_words - 1):
                embed = self.embedding(current_word).unsqueeze(0)  # 1 x 1 x E
                output_lstm, hidden = self.rnn(embed, hidden)  # 1 x 1 x H
                output = output_lstm[0]  # 1 x H
                scores = self.scoring(output)  # V
                _, current_word = torch.max(scores, dim=1)  # 1
                generated_words.append(current_word)
                if current_word[0].item() == self.stop:  # If end of line
                    break
        return torch.cat(generated_words, dim=0)
