# intall 
`pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0`
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

pip install nltk
pip install torchmetrics
pip install boto3
pip install filelock
pip install tokenizers
pip install dpu_utils
pip install h5py
pip install transformers
pip install ujson
pip install pynvml
pip install dgl
pip install gdown
pip install jsbeautifier


# 说明

  这个是SyseVr的复现，用pytorch重写了相关的文件，然后配置都是按文章里一致的，现在需要配置相同的bgru模型，然后使用相同的切片数据进行训练。
  
1. 数据集整理
    sysevr的使用的训练和验证数据集正好把原始切片数据集读取出来，然后copy。

2. 训练word2vec模型

3. bgru网络的数据预处理

4. bgru网络搭建
        