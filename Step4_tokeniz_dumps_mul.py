import torch,sys
import json
import ujson
import multiprocessing
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
import time
from torch.multiprocessing import Queue # 使用专门为torch设计的多进程队列
END_SIGNAL = "END_OF_PROCESSING"
num_workers_global = 40  # Or the number of cores you have
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

def process_data_line(data_line):
    try:
        code_snippet_1 = ujson.loads(data_line)
        code_snippet = filter_line_numbers(code_snippet_1)
        l = str(data_line).split("???")
        if len(l) < 3:
            return None
        if random() > 0.5:
            x1 = l[0]
            x2 = l[1]
        else:
            x1 = l[1]
            x2 = l[0]
        y = code_snippet.split("???")[2].split("\\")[0].strip()

        if Args.is_bpe_based == False:
            x1_idx = Vocab_my.transform(x1)
            x2_idx = Vocab_my.transform(x2)
        else:
            x1_idx = Vocab_bpe.tokens_to_indices(Vocab_bpe.subtokenize(x1))
            x2_idx = Vocab_bpe.tokens_to_indices(Vocab_bpe.subtokenize(x2))

        train_line = x1_idx + [int(Args.split_token)] + x2_idx + [int(Args.split_token)] + [int(y)]
        print("process one line!")
        # train_line = END_SIGNAL
        return torch.IntTensor(train_line).tolist() # 将张量转换为列表 序列化
    except:
        print("process error!")
        return None

def worker(input_queue, output_queue):
    while True:
        data_line = input_queue.get()
        if data_line == END_SIGNAL:
            output_queue.put(END_SIGNAL) #以None作为结束信号
            break
        output_queue.put(process_data_line(data_line))

def main():
    mode = sys.argv[1] # "train", "eval", "test"
    SRC_DIR = os.path.join(Args.data_pkl_path)
    SRC_FILE = os.path.join(SRC_DIR, Args.task_name + "_"+ f"{mode}.jsonl")
    DST_FILE = os.path.join(SRC_DIR)
    mmap_dataset_builder = MMapIndexedDatasetBuilder(DST_FILE + Args.task_name + "_mul_" + mode + ".mmap")

    # input_queue = multiprocessing.Queue()
    # output_queue = multiprocessing.Queue()
    input_queue = Queue() 
    output_queue = Queue()
    num_workers = num_workers_global  # Or the number of cores you have
    processes = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(input_queue, output_queue))
        p.start()
        processes.append(p)

    with open(SRC_FILE, 'rb') as file:
        for data_line in file:
            input_queue.put(data_line) # 同时读取、处理

    for _ in processes: # 向每个工作进程发送结束信号
        input_queue.put(END_SIGNAL)
    
    while True:
        tensor1 = output_queue.get()

        if tensor1 == END_SIGNAL:
            break
        tensor1 = torch.IntTensor(tensor1) # 将列表转换为张量 反序列化
        mmap_dataset_builder.add_item(tensor1)
        
        print("add line in mmap!")
        # break  # 测试
    
    mmap_dataset_builder.finalize(DST_FILE + Args.task_name + "_mul_" + mode + ".idx")
    print("index file down!")

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()