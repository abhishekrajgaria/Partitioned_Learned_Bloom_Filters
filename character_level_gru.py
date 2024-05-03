import torch.nn as nn

class CharacterLevelGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        super(CharacterLevelGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size,  batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        # print(embedded.shape)
        output, _ = self.gru(embedded)
        # print(output.shape)
        last_output = output[:, -1, :]
        # print(last_output.shape)
        output = self.fc(last_output) 
        # print(output.shape)
        output = self.sigmoid(output)
        # print(output.shape)
        return output