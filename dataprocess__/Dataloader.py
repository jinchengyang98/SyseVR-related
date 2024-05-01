from model.Trainingconfig import TrainingConfig as Args
from dataprocess.Dataset import SiaMese_Dataset

# 类实例化
Args = Args()

from ncc.tasks import register_task
from ncc.tasks.ncc_task import NccTask


@register_task(Args.task_name) # 注册任务
class DataLoader(NccTask): # 继承NccTask 
    def __init__(self, dictionary): # arg1：字典
        super(DataLoader, self).__init__(args=None)
        self.dictionary = dictionary # 使用的是BPE分词

    def load_dataset(self, split, data_file): # 加载数据集
        # define your loading rules # 定义你的加载规则
        from ncc.data.indexed_dataset import MMapIndexedDataset # 动态加载数据集
        from ncc.data.wrappers import TruncateDataset
        # truncate code with a length of 128 + 1 # 截断长度为128+1的代码
        dataset = TruncateDataset(
            MMapIndexedDataset(data_file), # arg1：mmap文件的路径
            truncation_length=Args.max_seq_length + 1, # 截断长度
        )
        datasizes = dataset.sizes # 保存数据集大小
        self.datasets[split] = SiaMese_Dataset(self.dictionary, dataset, datasizes) # 保存数据集