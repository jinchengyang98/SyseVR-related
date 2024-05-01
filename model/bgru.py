import torch
import torch.nn as nn

class BiGRUNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, num_layers, num_classes=2):
        super(BiGRUNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bigru = nn.GRU(embedding_dim, hidden_units, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_units, num_classes)  # 乘以2因为是双向的

    def forward(self, x):
        # Embedding
        x = self.embedding(x)

        # BiGRU
        out, _ = self.bigru(x)

        # 取最后时刻的输出
        out = out[:, -1, :]  # Using the output of the last timestep

        # Fully connected layer to classify
        output = self.fc(out)
        return output

# Example usage
# Assuming `vocab_size`, `embedding_dim`, `hidden_units`, `num_layers`, and `num_classes` are defined
# model = BiGRUNetwork(vocab_size, embedding_dim, hidden_units, num_layers, num_classes)
# outputs = model(torch.randint(0, vocab_size, (batch_size, sequence_length)))  # Random data example
