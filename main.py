import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
embedding_size = 200
hidden_size = 200
num_layers = 2
num_epochs = 20
batch_size = 20
sequence_length = 30
learning_rate = 1.0
dropout_prob = 0.5  

# Tokenizer
tokenizer = get_tokenizer('basic_english')

# Load data from local files
def load_data():
    data_path = 'ptb_data'
    with open(os.path.join(data_path, 'ptb.train.txt'), 'r') as f:
        train_data = f.read()
    with open(os.path.join(data_path, 'ptb.valid.txt'), 'r') as f:
        val_data = f.read()
    with open(os.path.join(data_path, 'ptb.test.txt'), 'r') as f:
        test_data = f.read()
    return train_data, val_data, test_data

# Build vocabulary
def build_vocab(data):
    tokens = tokenizer(data)
    counter = Counter(tokens)
    vocab = sorted(counter, key=counter.get, reverse=True)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

# Encode data
def encode_data(data, word2idx):
    tokens = tokenizer(data)
    encoded = [word2idx[word] for word in tokens if word in word2idx]
    return torch.tensor(encoded, dtype=torch.long)

# Custom Dataset
class PTBDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return (len(self.data) - 1) // self.seq_length

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        x = self.data[start:end]
        y = self.data[start+1:end+1]
        return x, y

# Model Definition
class RNNModel(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, num_layers, dropout_prob):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout_prob)
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_prob, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(output.size(0)*output.size(1), output.size(2)))
        return decoded, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device)

# Training function
def train_model(model, train_loader, criterion, optimizer, epoch, vocab_size):
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    for batch, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, targets.view(-1))
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()
        if batch % 100 == 0 and batch > 0:
            cur_loss = total_loss / 100
            ppl = np.exp(cur_loss)
            elapsed = time.time() - start_time
            print(f'| epoch {epoch} | {batch}/{len(train_loader)} batches | lr {learning_rate:.2f} | '
                  f'ms/batch {elapsed * 1000 / 100:.2f} | loss {cur_loss:.2f} | ppl {ppl:.2f}')
            total_loss = 0
            start_time = time.time()

# Evaluation function
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = criterion(output, targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Helper function to detach hidden states
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# Main function to run experiments
def main():
    train_data_raw, val_data_raw, test_data_raw = load_data()
    word2idx, idx2word = build_vocab(train_data_raw)
    vocab_size = len(word2idx)
    print(f'Vocab size: {vocab_size}')

    train_data = encode_data(train_data_raw, word2idx)
    val_data = encode_data(val_data_raw, word2idx)
    test_data = encode_data(test_data_raw, word2idx)

    train_dataset = PTBDataset(train_data, sequence_length)
    val_dataset = PTBDataset(val_data, sequence_length)
    test_dataset = PTBDataset(test_data, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Settings for experiments
    settings = [
        {'model_type': 'LSTM', 'dropout': 0.0, 'label': 'LSTM without Dropout'},
        {'model_type': 'LSTM', 'dropout': dropout_prob, 'label': 'LSTM with Dropout'},
        {'model_type': 'GRU',  'dropout': 0.0, 'label': 'GRU without Dropout'},
        {'model_type': 'GRU',  'dropout': dropout_prob, 'label': 'GRU with Dropout'},
    ]

    results = []

    for setting in settings:
        print(f"\nTraining {setting['label']}...")
        model = RNNModel(
            rnn_type=setting['model_type'],
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_prob=setting['dropout']
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)

        train_ppls = []
        val_ppls = []

        best_val_loss = None

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            train_model(model, train_loader, criterion, optimizer, epoch, vocab_size)
            train_loss = evaluate(model, train_loader, criterion)
            val_loss = evaluate(model, val_loader, criterion)
            train_ppl = np.exp(train_loss)
            val_ppl = np.exp(val_loss)
            train_ppls.append(train_ppl)
            val_ppls.append(val_ppl)
            print('-' * 89)
            print(f'| end of epoch {epoch} | time: {(time.time() - epoch_start_time):.2f}s | '
                  f'train loss {train_loss:.2f} | train ppl {train_ppl:.2f} | '
                  f'valid loss {val_loss:.2f} | valid ppl {val_ppl:.2f}')
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{setting['label']}_model.pt")
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                scheduler.step()

        # Load the best saved model.
        model.load_state_dict(torch.load(f"{setting['label']}_model.pt"))

        # Run on test data.
        test_loss = evaluate(model, test_loader, criterion)
        test_ppl = np.exp(test_loss)
        print('=' * 89)
        print(f'| End of training | test loss {test_loss:.2f} | test ppl {test_ppl:.2f}')
        print('=' * 89)

        results.append({
            'label': setting['label'],
            'learning_rate': learning_rate,
            'dropout': setting['dropout'],
            'train_ppl': train_ppl,
            'val_ppl': val_ppl,
            'test_ppl': test_ppl,
            'train_ppls': train_ppls,
            'val_ppls': val_ppls
        })

        # Plot convergence graph
        plt.figure()
        plt.plot(range(1, len(train_ppls)+1), train_ppls, label='Train Perplexity')
        plt.plot(range(1, len(val_ppls)+1), val_ppls, label='Validation Perplexity')
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity')
        plt.title(f'Perplexity vs Epochs for {setting["label"]}')
        plt.legend()
        plt.savefig(f'{setting["label"]}_perplexity.png')
        plt.show()

    # Summarize results in a table
    print('\nSummary of Results:')
    print(f'{"Model":<25} {"LR":<5} {"Dropout":<8} {"Train PPL":<10} {"Val PPL":<10} {"Test PPL":<10}')
    for res in results:
        print(f'{res["label"]:<25} {res["learning_rate"]:<5} {res["dropout"]:<8} '
              f'{res["train_ppl"]:<10.2f} {res["val_ppl"]:<10.2f} {res["test_ppl"]:<10.2f}')

if __name__ == '__main__':
    main()
