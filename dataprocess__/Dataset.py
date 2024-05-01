from model import Trainingconfig as Args
from ncc.data.ncc_dataset import NccDataset
Args = Args.TrainingConfig()
import torch
from dataprocess import my_Vocabulary as Vocab
Vocab = Vocab.VocabularyProcessor() # 类实例化
class SiaMese_Dataset(NccDataset):
    def __init__(self, dict, data, sizes):
        self.dict = dict
        self.data = data
        self.sizes = sizes
        self.Vocabbulary = Vocab
        self.pad = dict.pad()

    def __getitem__(self, index): # 这里定义数据的格式
        src_item = self.data[index] # .split(Args.split_token) # 先保证取到一个样本 FIXME 这里需要进行一定的分割方法
        # 1 分割

        src_item = self.split(src_item,torch.tensor(int(Args.split_token)))
        if len(src_item)<3 :
            return 
        # 2 截断或填充
        x1 = self.Vocabbulary.pad_or_truncate_sentence(src_item[0],Args.cut_length)

        x2 = self.Vocabbulary.pad_or_truncate_sentence(src_item[1],Args.cut_length)

        y = src_item[2]
        # x1 = src_item[0]
        # x2 = src_item[1]
        # y = src_item[2]
        example = {
            'id': index,
            'x1': x1,
            "x2": x2,
            'target': y,
        }
        return example

    def __len__(self):
        return len(self.data)
    

    def collate(self,samples, pad_idx): # 处理需要以批次形式加载数据，将来自数据集的一组样本转换为模型训练或评估所需的格式
        from ncc.data.tools import data_utils
        # def merge(key):
        #     return data_utils.collate_tokens( 
        #         [s[key] for s in samples],
        #         pad_idx, # 填充 
        #     )
        def merge(key):
            result = []
            for s in samples:
                if type(s) == 'NoneType' or s == None:
                    continue
                result.append(s[key])

            return data_utils.collate_tokens(  # 将samples中的key对应的值进行填充
                result,
                pad_idx, # 填充索引 
            )
        x1 = merge("x1")
        x2 = merge("x2")
        target = merge("target")
        # x1_tokens = merge('x1') # 将samples中的'source'对应的值进行填充
        # x2_tokens = merge("x2")
        # tgt_tokens = merge('target') # 将samples中的'target'对应的值进行填充
        return {
            'id': [s["id"] for s in samples if s!=None], # 将samples中的'id'对应的值进行填充
            'x1': x1, # 返回 将samples中的'source'对应的值进行填充的值
            "x2": x2,
            'target': target, # 返回 将samples中的'target'对应的值进行填充的值 
        }
    
    def collater(self, samples): # 将samples中的'source'、'target'、'id'对应的值进行填充
        return self.collate(samples, self.pad) # arg2：pad_idx
    
    def ordered_indices(self):# Return an ordered list of indices. Batches will be constructed based on this order.
        import numpy as np
        return np.random.permutation(len(self))

    def num_tokens(self, index): # Return the number of tokens in a sample.
        # Return the number of tokens in a sample.
        return self.sizes[index]
    
    def split(self, split_data,split_token): # split是tensor的格式
        """
        Split a tensor at each occurrence of the specified token.
        
        Args:
        tensor (torch.Tensor): The tensor to be split.
        token (int): The token used as a delimiter for splitting.

        Returns:
        list of torch.Tensor: A list of tensors, split at each occurrence of the token.
        """
        split_list = []
        start_idx = 0

        # Find all indices of the token
        indices = (split_data == split_token).nonzero(as_tuple=True)[0]

        for idx in indices:
            # Add the slice of tensor from the last token (or start) to the current token
            if idx > start_idx:
                split_list.append(split_data[start_idx:idx])
            start_idx = idx + 1

        # Add the remaining part of the tensor after the last token
        if start_idx < len(split_data):
            split_list.append(split_data[start_idx:])

        return split_list

    def size(self, index): 
        # Return an example's size.
        return self.sizes[index]