import os
from gensim.models import Word2Vec

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if fname.endswith('.txt'):
                file_path = os.path.join(self.dirname, fname)
                with open(file_path, 'r') as file:
                    for line in file:
                        # 处理每一行数据，去除空白字符后分割单词
                        words = line.strip().split()
                        if words:  # 确保不处理空行
                            yield words  # 使用 yield 生成每个单词列表

def train_model():
    sentences = MySentences('/home/deeplearning/nas-files/SyseVR-related/data/label_data/')  # 目录路径
    model = Word2Vec(min_count=5, vector_size=200, workers=40)
    # 由于 `sentences` 是一个迭代器，需要转换为列表来构建词汇表

    model.build_vocab(sentences)  # 一次性构建词汇表
    model.train(sentences, total_examples= model.corpus_count, epochs=model.epochs)  # 训练模型
    model.save('/home/deeplearning/nas-files/SyseVR-related/data/word2vec_model/word2vec_model.bin.gz')  # 保存模型

if __name__ == "__main__":
    train_model()
