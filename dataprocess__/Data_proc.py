import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as torchDataLoader
import os,pickle
import numpy as np
from random import random
from torch.utils.data import random_split
import dataprocess.my_Vocabulary as Vocab 
import model.Trainingconfig as Args
import json

Args = Args.TrainingConfig() # 类实例化

class Siamese_Dataset(Dataset):
    def __init__(self, dataset_path = Args.data_pkl_path,max_seq_length = Args.max_seq_length, char_based=False):
        self.dataset_path = dataset_path
        self.filepaths =  [os.path.join(dirpath, filename)
                        for dirpath, dirnames, filenames in os.walk(dataset_path)
                        for filename in filenames]

        self.max_seq_length = max_seq_length
        self.char_based = char_based
        self.vocab_path = Args.vocab_path
        self.indexes = self._create_indexes(self.dataset_path) # 为每个数据集文件创建索引
        self.vocab = Vocab.VocabularyProcessor(is_bpe_based=self.char_based) # 实例化词汇表

        # 构建词汇表
        if self.vocab.load_vocab(self.vocab_path):
            print('Vocabulary found!')
        else:
            print('Vocabulary not found! Building vocabulary...')
            self.vocab.build_vocab(self.filepaths)
            # 保存词汇表
            if Args.split_token not in self.vocab.word2idx:
                print("split_token not in vocab,add it")
                self.vocab.word2idx[Args.split_token] = self.vocab.num_words
                self.vocab.idx2word[self.vocab.num_words] = Args.split_token
            self.vocab.save_vocab(self.vocab_path)
        # 将样本分割词加入词表
        if Args.split_token not in self.vocab.word2idx:
            print("split_token not in vocab,add it")


    def __len__(self):
        return len(self.indexes)
    
    def _create_indexes(self,dataset_path): # 为每个数据集文件创建索引
        indexes = []
        current_position = 0

        # 检查 dataset_path 是否是字符串，如果是，则将其转换为单元素列表
        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]

        for filepath in dataset_path:
            with open(filepath, 'r') as file:  # 使用 'rb' 模式读取 pickle 文件
                # # pkl
                # data = pickle.load(file)
                # print(len(data))
                # # print(data)
                # for item in data:
                #     indexes.append(current_position)
                #     current_position += 1  # 每个样本增加一个索引

                for line in file:
                    indexes.append(current_position)
                    current_position += 1


        return indexes
    
    @staticmethod
    def my_batch_generator(self,dataset_path = Args.data_pkl_path, batch_size=Args.batch_size,task_mode = "train"):
        current_batch = []
            # 加载 .pkl 文件
        with open(dataset_path + "_"+task_mode + ".pkl", 'rb') as file:
            loaded_data = pickle.load(file)
            for load_data in loaded_data:
                current_batch.append(load_data)
                if len(current_batch) == batch_size:
                    yield current_batch # 动态加载一个batch的数据
                    current_batch = [] # TODO 需要考虑这个文件需要放在哪里 1.dataload需要划分train,eval,test数据集:使用了一个类提前对数据集进行划分 2.文件需要分batch加载:自定义dataload，然后使用yield关键字动态 
                                                # 3.需要按照孪生网络的输入格式进行加载
        if current_batch:
            yield current_batch  # 处理剩余的部分

    def __getitem__(self, idx):
        line_x = 0
        with open(self.dataset_path, 'r',encoding='utf-8') as file:
            count_idx = 0
            for line in file:
                if idx == count_idx:
                    line_x = line
                    break
                count_idx += 1
            
            # data = pickle.load(file) # pkl
            # data = json.load(file) # json
            # line = data[idx] #转移到指定的行
            # line = file.readline().strip()
            # print(line_x)
            l = line_x.strip().split("???") # 标签和句子之间用???分隔
            if len(l) < 3:
                return self.__getitem__((idx + 1) % len(self)) # 如果该行不符合要求，则读取下一行
            if random() > 0.5:
                x1 = l[0]
                x2 = l[1]
            else:
                x1 = l[1]
                x2 = l[0]
            # 分词并填充
            x1 = self.vocab.transform(x1)
            x2 = self.vocab.transform(x2)
            y = l[2].split("\\n")[0]
            print(y)
            # 这里可以添加对行的处理
        
        # line = torch.tensor(x1,dtype=torch.int64), torch.tensor(x2,dtype=torch.int64), torch.tensor(y,dtype= torch.int64) # 将行转换为tensor
        
        # return line
    
        return {
            'x1': torch.tensor(x1,dtype=torch.int64),
            'x2': torch.tensor(x2,dtype=torch.int64),
            'y': torch.tensor(int(y),dtype= torch.int64)  }
    
    # def __getitem__(self, idx):

        # x1, x2, y = [], [], []
        # all_line = []
        # filepath = self.filepaths[idx]
        # with open(filepath, 'r', encoding='ISO-8859-1') as f:
        #     while True:
        #         line = f.readline()
        #         line = line.strip()
        #         all_line.append(line)
        #         if not line:
        #             break
        # for line in all_line:

        #     l = line.strip().split("???") # 标签和句子之间用???分隔
        #     if len(l) < 3:
        #         continue

        #     if random() > 0.5:
        #         x1.append(l[0])
        #         x2.append(l[1])
        #     else:
        #         x1.append(l[1])
        #         x2.append(l[0])
            

        #     y.append(int(l[2]))
        # x1 = self.vocab.transform(x1)
        # x2 = self.vocab.transform(x2)
        # print(len(x1))
        # print(len(x2))
        # print(len(y))
        # return torch.tensor(x1,dtype=torch.int64), torch.tensor(x2,dtype=torch.int64), torch.tensor(y,dtype= torch.int64)

        # # 从文件中随机选择一行
        # random_line = random.choice(lines)
        # l = random_line.strip().split("???")
        # if len(l) < 3:
        #     return self.__getitem__((idx + 1) % len(self))

        # x1, x2, y = l[0].lower(), l[1].lower(), int(l[2])
        # return x1, x2, torch.tensor(y)
    # def __init__(self, Dateset_path, max_seq_length, char_based=False):
    #     self.Dataset_path = Dateset_path
    #     self.max_seq_length = max_seq_length
    #     self.char_based = char_based
    #     self.x1, self.x2, self.y = self.load_data()

    # def load_data(self):
    #     x1, x2, y = [], [], []

    #     for dirpath, dirnames, filenames in os.walk(self.Dataset_path):

    #         # 打印所有文件
    #         for filename in filenames:
    #             filepath = os.path.join(dirpath, filename) 
    #             with open(os.path.join(filepath), 'r',encoding='ISO-8859-1') as f:
    #                 for line in f:
    #                     l = line.strip().split("???") # 标签和句子之间用???分隔
    #                     if len(l) < 2:
    #                         continue
    #                     if random() > 0.5:
    #                         x1.append(l[0].lower())
    #                         x2.append(l[1].lower())
    #                     else:
    #                         x1.append(l[1].lower())
    #                         x2.append(l[0].lower())
    #                     y.append(int(l[2]))
    #     return np.array(x1), np.array(x2), np.array(y)

    # def __len__(self):
    #     return len(self.y)

    # def __getitem__(self, idx):
    #     return {
    #         'x1': self.x1[idx],
    #         'x2': self.x2[idx],
    #         'y': self.y[idx]
    #     }
    
class TextPairTestDataset(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.x1, self.x2, self.y = self.load_test_data()

    def load_test_data(self):
        x1, x2, y = [], [], []
        for fname in os.listdir(self.filepath):
            with open(os.path.join(self.filepath, fname), 'r') as f:
                for line in f:
                    l = line.strip().split("\t")
                    if len(l) < 3:
                        continue
                    x1.append(l[1].lower())
                    x2.append(l[2].lower())
                    y.append(int(l[0]))
        return np.array(x1), np.array(x2), np.array(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'x1': self.x1[idx],
            'x2': self.x2[idx],
            'y': self.y[idx]
        }

