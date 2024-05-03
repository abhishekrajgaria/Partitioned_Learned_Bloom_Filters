import torch.nn as nn

class CharacterLevelGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        super(CharacterLevelGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        print(embedded.shape())
        output, _ = self.gru(embedded.view(len(x), 1, -1))
        output = self.fc(output.view(len(x), -1))
        output = self.sigmoid(output) 
        return output