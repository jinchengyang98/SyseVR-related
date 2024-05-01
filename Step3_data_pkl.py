import pickle,os
import config.Trainingconfig as Args
import os
import random
import pickle
from dataprocess.Data_save import Data_save
# 类实例化
Data_saver = Data_save()
Args = Args.TrainingConfig()


if __name__ == '__main__':

    # 加载所有数据集文件的路径
    filepaths =  [os.path.join(dirpath, filename)
                        for dirpath, dirnames, filenames in os.walk(Args.data_path)
                        for filename in filenames]

    if os.path.exists(Args.data_pkl_path)!=True:
        os.mkdir(Args.data_pkl_path)
    print("filepath size:",len(filepaths))
    Data_saver.process_pkldata(filepaths)
