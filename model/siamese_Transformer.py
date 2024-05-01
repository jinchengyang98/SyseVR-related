import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
from dataprocess.Data_proc import Siamese_Dataset, TextPairTestDataset
from config.Trainingconfig import TrainingConfig as Args
from ncc.models import register_model
# 假设 Args 和 register_model 已经定义
Args = Args() # 类实例化
@register_model(Args.task_name + '_Transformer')
class Demo_SiameseTransformer(NccLanguageModel):
    def __init__(self, dictionary, decoder):
        super().__init__(decoder)
        self.dictionary = dictionary

    @classmethod
    def build_model(cls, dictionary):
        class Decoder(NccDecoder):
            def __init__(self, dictionary, batch_size=Args.batch_size, hidden_units=Args.Transformer_hidden_units, 
                         embedding_size=Args.Transformer_embedding_dim, num_layers=Args.Transformer_num_layers, 
                         vocab_size=len(dictionary), nhead=Args.Transformer_heads):
                super(Decoder, self).__init__(dictionary)

                self.batch_size = batch_size
                self.hidden_units = hidden_units
                self.num_layers = num_layers
                
                # Embedding layer
                self.embedding = nn.Embedding(vocab_size, embedding_size)

                # Positional Encoding
                self.pos_encoder = PositionalEncoding(embedding_size)

                # Transformer Encoder Layer
                encoder_layers = TransformerEncoderLayer(embedding_size, nhead, hidden_units)
                self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

                # 分类层
                self.classifier = nn.Linear(in_features=embedding_size, out_features=Args.num_classes)
            
            def forward(self, x):
                # Embedding
                x = self.embedding(x)

                # Positional Encoding
                x = self.pos_encoder(x)

                # Transformer
                x = self.transformer_encoder(x)
                x = x.mean(dim=1) # 可以使用均值或其他汇聚操作

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
        x = self.decoder.classifier(distance)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create constant 'pe' matrix with values dependent on 
        # position and d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding
        x = x + self.pe[:x.size(0), :]
        return x
