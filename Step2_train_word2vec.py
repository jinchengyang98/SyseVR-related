from gensim.models import Word2Vec
from dataprocess import my_Vocabulary as Vocab_my
import os
from config.Trainingconfig import TrainingConfig as Args
Args = Args() # 类实例化
Vocab_golbal = Vocab_my.VocabularyProcessor(is_bpe_based=Args.is_bpe_based,Tasks_name=Args.task_name) # is bpe based ==True是为了使用voc加载词汇进行分词
class MySentences(object):
    def __init__(self, dirname,vocab):
        self.dirname = dirname
        self.vocab = vocab

    def __iter__(self): # 保证数据集的读取是可迭代的
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='ISO-8859-1'):
                line = self.vocab.tokenize_zz(line) # 使用正则化分词
                yield line
    

sentences = MySentences('/home/deeplearning/nas-files/SyseVR-related/data',vocab = Vocab_golbal)  # 一个包含多个代码文件的目录
model = Word2Vec(sentences)
print("Traing word2vec!")
# 训练模型
model = Word2Vec(sentences, vector_size=100, window=50, min_count=5, workers=16)
# v1 vector_size=100, window=50, min_count=5, workers=16
# v2 vector_size=100, window=50, min_count=5, workers=16 加入了未知词的处理 《UNK

# 保存模型
model.save("/home/deeplearning/nas-files/SyseVR-related/model/word2vec_v2.model")
print("Trained word2vec and saved model")
