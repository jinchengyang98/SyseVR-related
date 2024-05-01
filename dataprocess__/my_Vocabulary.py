# import nltk
import torch
from collections import Counter
import re
import pickle
import os
from model.Trainingconfig import TrainingConfig as Args
from dataprocess import help as helper
from gensim.models.word2vec import Word2Vec
Args = Args() # 类实例化
#from ncc.data.dictionary import TransformersDictionary
# re分词，一段中有除了数字和字母以外的其他字符，则以这些字符为分隔符
TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)

# 预训练分词
pretrain_path = os.path.join(Args.root_data_path, "pretrain/")
#vocab = TransformersDictionary.from_pretrained(pretrain_path + "microsoft/codebert-base", do_lower_case=False) # arg1：预训练模型的名称，arg2：是否小写转换


def tokenizer_char(iterator):
    for value in iterator:
        yield list(value)

def tokenizer_word(iterator):
    for value in iterator:
        yield TOKENIZER_RE.findall(value)# 

class VocabularyProcessor:

    def __init__(self, is_bpe_based=False,Tasks_name = Args.task_name):
        self.task_name = Tasks_name
        self.word2idx = {}
        self.idx2word = {}
        self.is_bpe_based = is_bpe_based
        self.num_words = 0
        self.vocab = ""
        self.vocab_path = Args.vocab_path
        self.train_data_path = Args.data_path

        self.word2idx['<PAD>'] = 0  # Padding Token
        self.idx2word[0] = '<PAD>'
        # unk也处理为0
        self.word2idx['<UNK>'] = 0

        # 建立词表
        if self.load_vocab(self.vocab_path)==True or self.is_bpe_based==True:
            print('Vocabulary found!')
        else:# 词表不存在
            print('Vocabulary not found! Building vocabulary...')
            self.build_vocab(self.train_data_path)
            self.save_vocab(self.vocab_path) # 保存词表
            #self.save_vocab(vocab_path = '/home/deeplearning/nas-files/naturalcc/data/train_snli/test_file')
        
    def tokenize_bpe(self,sentence): # BPE分词
        # return nltk.word_tokenize(text)
        # print(type(text))
        # return TOKENIZER_RE.findall(text)
        #return helper.filter_line_numbers(self.vocab.subtokenize(sentence)) #删除其中的行号
        return False
    
    def tokenize_zz(self,sentence): # 正则分词
        TOKENIZER_RE.findall(sentence)
        # 删除行号
        sentence = helper.filter_line_numbers(sentence)
        return TOKENIZER_RE.findall(sentence)

    # 更新词表
    def update_vocab(self, text):
        tokens = self.tokenize_zz(text)
        for token in tokens:
            if token not in self.word2idx:
                self.word2idx[token] = self.num_words
                self.idx2word[self.num_words] = token
                self.num_words += 1

    # 建立词表
    def build_vocab(self,filepaths, min_frequency=5): 
        # 读取数据集的路径
        for root ,dirs,files in os.walk(filepaths):
            for file in files:
                if file.endswith(".txt"):
                    filepath = os.path.join(root,file)
                    with open(filepath, 'r', encoding='ISO-8859-1') as f:
                        for line in f:
                            # 去除其中的行号
                            line = helper.filter_line_numbers(line)
                            if self.is_bpe_based:
                                l = self.tokenize_bpe(line)
                            else:
                                l = self.tokenize_zz(line)
                            # 更新词表
                            self.update_vocab(l[0].lower())
        print("Vocabulary build Success!")
               
    def transform(self, sentence, max_seq_length=Args.cut_length): # 分词、截断、填充
        # 分词  
        if self.is_bpe_based:
            tokens = self.tokenize_bpe(sentence)
            word_ids = [self.wordtoidx(token) for token in tokens]  # 0 is the index for <PAD>
        else:
            tokens = self.tokenize_zz(sentence)
            # 加载word2vec
            model = Word2Vec.load(Args.word2vec_vocab_path)
            # 转换词
            # word_ids = [model.wv[token] if token in model.wv else 0 for token in tokens]
            word_to_index = {word: i for i, word in enumerate(model.wv.index_to_key)}
            word_ids = [word_to_index[token] if token in word_to_index else 0 for token in tokens]
        # 截断或填充序列
        # tokens = self.pad_or_truncate_sentence(x_sentence = tokens)
        
        # 截断
        word_ids = word_ids[:max_seq_length]
        # 填充
        word_ids += [0] * (max_seq_length - len(word_ids)) # 用0填充
        return word_ids
    
    def wordtoidx(self,token): # 将单个token转换为索引
        token_idx = self.word2idx.get(token)
        return token_idx
    
    def token_to_idx(self,token_list): # 将token序列转换为索引
        tokens_idx = self.vocab.tokens_to_indices(token_list)
        return tokens_idx

    # 将词汇表保存到文件
    def save_vocab(self, vocab_path):
        # 检查目录是否存在
        if not os.path.exists(vocab_path):
            os.makedirs(vocab_path)
        # 保存词汇表
        with open(vocab_path + '_vocab.pkl' , 'wb') as f:
            pickle.dump(self.word2idx, f)
            pickle.dump(self.idx2word, f)
            pickle.dump(self.num_words, f)
    
    def load_vocab(self, vocab_path):
        # 检查文件是否存在
        if not os.path.exists(vocab_path + self.task_name + '_vocab.pkl'):
            print("Vocab file doesn't exist")
            return False
        else:
             # 读取词汇表
            with open(vocab_path + self.task_name + '_vocab.pkl', 'rb') as f:
                self.word2idx = pickle.load(f)
                self.idx2word = pickle.load(f)
                self.num_words = pickle.load(f)
            return True
    
    # 截断或填充序列
    def pad_or_truncate_sentence(self, x_sentence, max_seq_length=Args.cut_length):
        if len(x_sentence) > max_seq_length:
            x_sentence = x_sentence[:max_seq_length] #tensor可以以这样的方式进行截断
        else:
            # print("x_sentence:",type(x_sentence))
            # print("pad:",type(self.idx2word[0]))
            pad_tensor = torch.full((max_seq_length - len(x_sentence),), self.word2idx['<PAD>'], dtype=torch.int)
            x_sentence = torch.cat((x_sentence, pad_tensor))
            # x_sentence += torch.IntTensor(self.word2idx['<PAD>']) * (max_seq_length - len(x_sentence)) # 填充是在序列化之前，所以使用<pad>的填充
        # print("x_sentence:",len(x_sentence))
        return x_sentence




