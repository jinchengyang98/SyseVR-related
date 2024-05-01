import torch,sys
import json
import ujson
from tqdm import tqdm
sys.path.append('/home/deeplearning/nas-files/naturalcc/src/')
from ncc.data.indexed_dataset import MMapIndexedDatasetBuilder
from config.Trainingconfig import TrainingConfig as Args
import os,pickle
Args = Args() # 类实例化
from ncc.data.dictionary import TransformersDictionary
from dataprocess.Data_proc import Siamese_Dataset as Dataset
from dataprocess import my_Vocabulary as Vocab_my
from random import random

# #bpe dictionary
pretrain_path = os.path.join(Args.root_data_path, "pretrain/") 
Vocab_bpe = TransformersDictionary.from_pretrained(pretrain_path + "microsoft/codebert-base", do_lower_case=False) # arg1：预训练模型的名称，arg2：是否小写转换
Vocab_my = Vocab_my.VocabularyProcessor(is_bpe_based=Args.is_bpe_based,Tasks_name = Args.task_name)

def total_lines(reader):
    num = sum(1 for _ in reader)
    # reader.seek(0)
    return num

def filter_line_numbers(text, special_marker='???'):# 这个函数的功能是为了过滤配对数据中的行号
    import re
    # 按空格分割文本
    parts = text.split(' ')
    filtered_parts = []

    for part in parts:
        # 检查是否包含特殊标记
        if special_marker in part:
            filtered_parts.append(part)
        else:
            # 移除所有仅由数字组成的部分
            if not re.fullmatch(r'\d+', part):
                filtered_parts.append(part)

    # 重新组合处理过的部分
    filtered_text = ' '.join(filtered_parts)

    return filtered_text

if __name__ == '__main__':
    mode = sys.argv[1] # "train", "eval", "test"
    SRC_DIR = os.path.join(Args.data_pkl_path)
    # for mode in ["train", "eval", "test"]:
    SRC_FILE = os.path.join(SRC_DIR,Args.task_name + "_"+ f"{mode}.jsonl")
    DST_FILE = os.path.join(SRC_DIR)
    mmap_dataset_builder = MMapIndexedDatasetBuilder(DST_FILE + Args.task_name + "_" +mode + ".mmap") # arg1：mmap文件的路径
    x1, x2, y = [], [], []
    with open(SRC_FILE, 'rb') as file: # 读取src文件
        count = 0
        print("Processing file:",SRC_FILE)
        # dataset = pickle.load(file) #
        # for data_line in enumerate(tqdm(dataset,total = total_lines(dataset))):
        for data_line in file:
            # 分词
            # print(data_line)  
            code_snippet_1 = ujson.loads(data_line)
            code_snippet = filter_line_numbers(code_snippet_1) # 过滤数据中的行号
            count += 1
            print("Processing line:",count) # jsonl
            l = str(data_line).split("???")
            if len(l)<3:
                continue
            if random()>0.5:
                x1 = l[0]
                x2 = l[1]
            else:
                x1 = l[1]
                x2 = l[0]

            y =  code_snippet_1.split("???")[2].split("\\")[0].strip() # l[2].split("\\")[0].strip()
            # # 分词
            # x1_bpe = Vocab_bpe.subtokenize(x1)
            # x2_bpe = Vocab_bpe.subtokenize(x2)
            # # 转化为索引
            # x1_idx = Vocab_bpe.tokens_to_indices(x1_bpe)
            # x2_idx = Vocab_bpe.tokens_to_indices(x2_bpe)
            if Args.is_bpe_based == False:
                x1_idx = Vocab_my.transform(x1)
                x2_idx = Vocab_my.transform(x2)
            else:
                x1_idx = Vocab_bpe.tokens_to_indices(Vocab_bpe.subtokenize(x1))
                x2_idx = Vocab_bpe.tokens_to_indices(Vocab_bpe.subtokenize(x2))
            # if len(x1_bpe) > Args.max_seq_length or len(x2_bpe) > Args.max_seq_length: # 有的长度不符合就跳过该样本
            #     continue

            # print(x1_bpe)
            # print(y)
            # print(x2_bpe)
            # raw_code_tokens = data_line
            # after_code_tokens = vocab.subtokenize(raw_code_tokens) # 将代码片段进行BPE处理
            train_line = x1_idx + [int(Args.split_token)] + x2_idx + [int(Args.split_token)] + [int(y)] # 组合BPE处理后的代码片段
            # print(train_bpe_line)

            # 转化为tensor
            tensor1 = torch.IntTensor(train_line) # 将BPE处理后的代码片段转换为tensor
            # tensor_split = torch.split(tensor1,Args.split_token)
            mmap_dataset_builder.add_item(tensor1) # 将tensor添加到mmap文件中 内存映射文件
    mmap_dataset_builder.finalize(DST_FILE + Args.task_name + "_" +mode + ".idx") # arg1：idx文件的路径 索引文件