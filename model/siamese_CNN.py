import torch
import torch.nn as nn
import math
from ncc.models import register_model
from config.Trainingconfig import TrainingConfig as Args
from ncc.models.ncc_model import NccLanguageModel
from ncc.modules.decoders.ncc_decoder import NccDecoder

# 假设 Args 和 register_model 已经定义
Args = Args()  # 类实例化
@register_model(Args.task_name + '_CNN')
class Demo_SiameseCNN(NccLanguageModel):
    def __init__(self, dictionary, decoder):
        super().__init__(decoder)
        self.dictionary = dictionary

    @classmethod
    def build_model(cls, dictionary):
        class Decoder(NccDecoder):
            def __init__(self, dictionary, batch_size=Args.batch_size, 
                         embedding_size=Args.CNN_embedding_dim, num_layers=Args.CNN_num_layers,
                         vocab_size=len(dictionary), num_channels=Args.CNN_num_channels,CNN_kernel_size=Args.CNN_kernel_size):
                super(Decoder, self).__init__(dictionary)

                self.embedding = nn.Embedding(vocab_size, embedding_size)

                # CNN layers
                self.convs = nn.ModuleList([
                    nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=ks)
                    for ks in CNN_kernel_size
                ])

                # Max pooling
                self.pool = nn.MaxPool1d(2)

                # Fully connected layer
                self.fc = nn.Linear(num_channels * len(self.convs), Args.num_classes)

            def forward(self, x):
                x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
                x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]

                # Apply convolutions and pooling
                x = [self.pool(torch.relu(conv(x))) for conv in self.convs]
                x = torch.cat(x, dim=2)  # Concatenate in the last dim

                # Flatten and pass through the linear layer
                x = x.view(x.size(0), -1)
                x = self.fc(x)

                return x

        decoder = Decoder(dictionary)
        return cls(dictionary, decoder=decoder)

    def forward(self, input_x1, input_x2):
        output1 = self.decoder(input_x1)
        output2 = self.decoder(input_x2)

        # Calculate the distance
        distance = torch.sqrt(torch.sum(torch.pow(output1 - output2, 2), 1, keepdim=True))
        distance = torch.div(distance, torch.add(torch.sqrt(torch.sum(torch.pow(output1, 2), 1, keepdim=True)), 
                                                 torch.sqrt(torch.sum(torch.pow(output2, 2), 1, keepdim=True))))
        x = self.decoder.fc(distance)
        return x
