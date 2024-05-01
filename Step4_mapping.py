import torch
import json
from tqdm import tqdm
from ncc.data.indexed_dataset import MMapIndexedDatasetBuilder
from config.Trainingconfig import TrainingConfig as Args
import os,pickle
Args = Args() # 类实例化
from ncc.data.dictionary import TransformersDictionary
from dataprocess.Data_proc import Siamese_Dataset as Dataset
from random import random
import dataprocess.my_Vocabulary as Vocab
Vocab = Vocab.VocabularyProcessor() # 类实例化
# 建立词汇表
Vocab.build_vocab(Args.vocab_path)
# #bpe dictionary
pretrain_path = os.path.join(Args.root_data_path, "pretrain/") 

vocab = TransformersDictionary.from_pretrained(pretrain_path + "microsoft/codebert-base", do_lower_case=False) # arg1：预训练模型的名称，arg2：是否小写转换


### 这个文件的主要目的是为了给数据集的json文件建立索引，然将索引存储在mmap文件中,作用有点像dataset和dataload的结合体


def total_lines(reader):
    num = sum(1 for _ in reader)
    # reader.seek(0)
    print ( num)
    return num


SRC_DIR = os.path.join(Args.data_pkl_path)
for mode in ["train", "eval", "test"]:
    SRC_FILE = os.path.join(SRC_DIR,Args.task_name + "_"+ f"{mode}.json")
    DST_FILE = os.path.join(SRC_DIR)
    mmap_dataset_builder = MMapIndexedDatasetBuilder(DST_FILE + Args.task_name + "_" +mode + ".mmap") # arg1：mmap文件的路径
    x1, x2, y ,dataset= [], [], [],[]
    with open(SRC_FILE, 'r') as file: # 读取src文件

        for line in file:
            dataset.append(line)
        for data_line in enumerate(tqdm(dataset,total = total_lines(dataset))):
            # 分词
            # print(data_line)
            
            l = str(data_line).split(Args.split_token)
            if len(l)<3:
                continue
            if random()>0.5:
                x1 = l[0]
                x2 = l[1]
            else:
                x1 = l[1]
                x2 = l[0]
            y = l[2]
            # BPE处理
            x1_bpe = Vocab.transform(x1)
            x2_bpe = Vocab.transform(x2)

            # raw_code_tokens = data_line
            # after_code_tokens = vocab.subtokenize(raw_code_tokens) # 将代码片段进行BPE处理
            train_bpe_line = x1_bpe + [Vocab.wordtoidx(Args.split_token)] + x2_bpe + [Vocab.wordtoidx(Args.split_token)] + [str(y)] # 组合BPE处理后的代码片段
            # 转化为tensor
            tensor = torch.IntTensor(train_bpe_line) # 将分词后的代码片段转换为tensor
            mmap_dataset_builder.add_item(tensor) # 将tensor添加到mmap文件中
    mmap_dataset_builder.finalize(DST_FILE + Args.task_name + "_" +mode + ".idx") # arg1：idx文件的路径