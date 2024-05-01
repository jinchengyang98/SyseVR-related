#! /usr/bin/env python
#coding=utf-8
import torch
import torch.nn as nn
from dataprocess.Data_proc import Siamese_Dataset, TextPairTestDataset
from config.Trainingconfig import TrainingConfig as Args
Args = Args() # 类实例化

class SiameseLSTM(nn.Module):
    def __init__(self, batch_size, hidden_units, vocab_size, embedding_size,num_layers=3):
        super(SiameseLSTM, self).__init__()

        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.num_layers = 3
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_size, hidden_units, num_layers=self.num_layers, 
                            bidirectional=True, batch_first=True, dropout=0.5)

        # 添加一个全连接层，例如，如果您的任务是二分类，可以这样初始化：
        self.classifier = nn.Linear(in_features=1, out_features=2)

    def forward_once(self, x):
        # Embedding
        x = self.embedding(x)

        # BiLSTM
        outputs, _ = self.lstm(x)
        outputs = outputs[:, -1, :]

        return outputs

    def forward(self, input_x1, input_x2):
        # Pass through the LSTM layers
        output1 = self.forward_once(input_x1)
        # print("distance:",input_x1.shape)
        output2 = self.forward_once(input_x2)
        # print("distance:",input_x2.shape)

        # Calculate the distance
        distance = torch.sqrt(torch.sum(torch.pow(output1 - output2, 2), 1, keepdim=True))
        distance = torch.div(distance, torch.add(torch.sqrt(torch.sum(torch.pow(output1, 2), 1, keepdim=True)), 
                                                 torch.sqrt(torch.sum(torch.pow(output2, 2), 1, keepdim=True))))
        # print("distance:",distance.shape)
        # distance = torch.reshape(distance, [-1])
        # print("distance:",distance.shape)
        x = self.classifier(distance) # 线性层 2分类，单值输出
        x = torch.softmax(x,dim = 1)  # 应用Sigmoid激活函数
        # print("x",x)
        return x

    def contrastive_loss(self, y, d):
        tmp = y * torch.pow(d, 2)
        tmp2 = (1 - y) * torch.pow(torch.clamp(1 - d, min=0.0), 2)
        return torch.sum(tmp + tmp2) / self.batch_size / 2
    
    def train_model(self, train_loader, optimizer, criterion, epoch, device):
        self.train()
        total_loss = 0
        for batch in train_loader:
            x1, x2, y = batch['x1'].to(device), batch['x2'].to(device), batch['y'].to(device)

            optimizer.zero_grad()
            output = self.forward(x1, x2)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, total_loss / len(train_loader)))

    def eval_step(self, dev_loader, device):
        self.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for batch in dev_loader:
                x1, x2, y = batch['x1'].to(device), batch['x2'].to(device), batch['y'].to(device)
                output = self.forward(x1, x2)
                # 此处根据模型的输出和标签计算准确度等指标
                # total_acc += (output.argmax(1) == y).sum().item()
                # total_count += y.size(0)
        # return total_acc / total_count


from ncc.models import register_model
from ncc.modules.base.layers import (
    Embedding, Linear, LSTM
)
from ncc.models.ncc_model import NccLanguageModel

@register_model(Args.task_name) # 注册模型
class Demo_SiameseLSTM(NccLanguageModel):
    def __init__(self, dictionary, decoder):
        super().__init__(decoder)
        self.dictionary = dictionary

    @classmethod
    def build_model(cls, dictionary):
        from ncc.modules.decoders.ncc_decoder import NccDecoder

        class Decoder(NccDecoder):                
            def __init__(self, dictionary,batch_size = Args.batch_size, hidden_units = Args.lstm_hidden_units, embedding_size = Args.embedding_dim,num_layers=Args.lstm_num_layers,vocab_size = len(dictionary)):
                super(Decoder, self).__init__(dictionary) # arg1：字典

                self.batch_size = batch_size
                self.hidden_units = hidden_units
                self.num_layers = num_layers
                
                # Embedding layer
                self.embedding = nn.Embedding(vocab_size, embedding_size)

                # LSTM layer
                self.lstm = nn.LSTM(embedding_size, hidden_units, num_layers=self.num_layers, 
                                    bidirectional=True, batch_first=True, dropout=Args.dropout_rate)

                # 添加一个全连接层，例如，如果您的任务是二分类，可以这样初始化：
                self.classifier = nn.Linear(in_features=1, out_features=Args.num_classes) # 定义分类器
            
            def forward(self, x):
                # Embedding
                x = self.embedding(x)

                # BiLSTM
                outputs, _ = self.lstm(x)
                outputs = outputs[:, -1, :]
                return outputs

        decoder = Decoder(dictionary)
        return cls(dictionary, decoder=decoder)

    def forward(self, input_x1, input_x2):
        # Pass through the LSTM layers
        output1 = self.decoder.forward(input_x1)
        # print("distance:",input_x1.shape)
        output2 = self.decoder.forward(input_x2)
        # print("distance:",input_x2.shape)

        # Calculate the distance 计算距离 
        distance = torch.sqrt(torch.sum(torch.pow(output1 - output2, 2), 1, keepdim=True))
        distance = torch.div(distance, torch.add(torch.sqrt(torch.sum(torch.pow(output1, 2), 1, keepdim=True)), 
                                                 torch.sqrt(torch.sum(torch.pow(output2, 2), 1, keepdim=True))))
        # 线性层 2分类，单值输出
        x = self.decoder.classifier(distance)
        # x = torch.softmax(x,dim = 1) # 查看损失函数是否会自动计算loss
        return x

