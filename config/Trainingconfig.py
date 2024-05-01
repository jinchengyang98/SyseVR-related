import torch 
import time 
class TrainingConfig:
    def __init__(self):


        # Task
        self.task_name = 'SyseVR_task'
        self.task_mode = ["train"] # "train", ,"test","train"

        # 模型参数
        ## LSTM 
        self.embedding_dim = 100            # 嵌入层的维度
        self.lstm_hidden_units = 20         # LSTM层隐藏单元数
        self.dropout_rate = 0.5             # Dropout层的丢弃率
        self.lstm_num_layers = 10           # LSTM层数
        ## Transformer
        self.Transformer_embedding_dim = 512        # 嵌入层的维度 
        self.Transformer_hidden_units = 20          # Transformer层隐藏单元数
        self.Transformer_num_layers = 10            # Transformer层数
        self.Transformer_heads = 8                  #多头注意力机制中头的数量，必须能被Transformer_embedding_dim整除;如果 Transformer_embedding_dim 是 512，nhead可以是 8
        self.Transformer_dim_feedforward = 1024     # dim_feedforward：前馈网络的维度，通常是 Transformer_embedding_dim 的两倍，例如 512的Transformer_embedding_dim 对应 1024 的 dim_feedforward。
        ## CNN
        self.CNN_embedding_dim = 100         # 嵌入层的维度
        self.CNN_num_layers = 10             # CNN层数
        self.CNN_num_channels = 10           # CNN层通道数
        self.CNN_kernel_size = 3             # CNN层卷积核大小
        self.num_classes = 2         # 类别数
        self.char_based = False      # 是否使用字符级别的模型

        # 训练参数
        self.batch_size = 128       # 批处理大小
        self.num_epochs = 100       # 训练的总轮数
        self.learning_rate = 0.001  # 学习率
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #BGRU
        self.BGRU_embedding_dim = 200         # 嵌入层的维度
        self.BGRU_hidden_units = 50          # LSTM层隐藏单元数
        self.BGRU_num_layers = 10              # LSTM层数
        self.BGRU_num_classes = 2              # 类别数


        # 数据集参数
        self.max_seq_length = 1000  # 序列的最大长度
        self.cut_length = 1000        # 截断长度
        self.min_word_freq = 5        # 词频小于该值的词将被过滤 
        self.train_ratio = 0.8       # 训练集比例
        self.dev_ratio = 0.1         # 验证集比例
        self.test_ratio = 0.1        # 测试集比例
        self.num_load_worker = 4     # dataloader加载数据的线程数
        self.is_shuffle = True       # 是否打乱数据集
        self.task_type = "binary"    # ["binary", "multiclass", "multilabel"]
        self.split_token = "1111000"     # 数据集x分隔符


        # 路径参数
        self.root_data_path = '/home/deeplearning/nas-files/SyseVR-related/data/'             # 数据集根目录
        self.slice_data_path = '/home/deeplearning/nas-files/SyseVR-related/data/label_data/'  # 切片数据集根目录
        # 模型保存路径
        self.save_path = self.root_data_path+'runs/model.pth'

        # self.data_path = self.root_data_path + 'train_snli/'                             # 数据存放路径
        self.data_path = "/home/deeplearning/nas-files/SyseVR-related/data"
        self.run_save_path = self.root_data_path + 'runs/' + self.task_name + '/'        # 运行日志保存路径
        self.data_pkl_path = self.root_data_path + self.task_name + "/"                  # 数据集pkl文件存放路径
        
        # 词汇相关
        self.vocab_path = self.root_data_path + 'vocab/'  # + self.task_name + '_vocab.pkl'
        self.is_bpe_based = False
        self.word2vec_vocab_path = self.root_data_path + 'word2vec_model/own2_word2vecmodel.bin'
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.model_save_path = self.root_data_path + str(self.task_mode) + timestamp +'_model.pth'
        
        self.SYS_FUNC_MAPPING_PATH = '/home/deeplearning/nas-files/SyseVR-related/dataprocess/function.xls'

        # 日志相关

        self.log_path = "/home/deeplearning/nas-files/SyseVR-related/data/Logs/"


# 使用示例
args = TrainingConfig()

# 然后您可以在代码中这样使用这些参数：
# 例如，在定义模型时：
# model = MyModel(args.embedding_dim, args.hidden_units)

# 在定义 DataLoader 时：
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
