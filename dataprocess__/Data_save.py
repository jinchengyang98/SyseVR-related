import pickle,os
import model.Trainingconfig as Args
import os
import random
import pickle
import json

Args = Args.TrainingConfig() # 类实例化

class Data_save:
    def __init__(self) -> None:
        pass

    def save_to_pkl(self,data, filepath):
        # if not os.path.exists(filepath):
        # # PKL
        # # 使用写入模式创建新文件
        #     with open(filepath, 'wb') as file:
        #         pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        # else:
        #     # 文件已存在，使用追加模式
        #     with open(filepath, 'ab') as file:
        #         pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # JSON
        with open(filepath, 'a', encoding='utf-8') as file:
            for item in data:
                json_string = json.dumps(item)
                file.write(json_string + '\n')



    def split_and_save_content(self,filepath, train_ratio, val_ratio, output_dir,count = 0):
        # 读取文件内容
        with open(filepath, 'r',encoding='ISO-8859-1') as file:
            lines = file.readlines()

        # 打乱数据顺序
        random.shuffle(lines)

        # 计算分割点
        total_lines = len(lines)
        train_end = int(total_lines * train_ratio)
        val_end = train_end + int(total_lines * val_ratio)

        # 分割数据集
        train_data = lines[:train_end]
        val_data = lines[train_end:val_end]
        test_data = lines[val_end:]

        # 保存到不同模式的 .pkl 文件
        
        self.save_to_pkl(train_data, os.path.join(output_dir, f"{Args.task_name}_train1.jsonl")) # .pkl
        self.save_to_pkl(val_data, os.path.join(output_dir, f"{Args.task_name}_eval1.jsonl")) # .pkl
        self.save_to_pkl(test_data, os.path.join(output_dir, f"{Args.task_name}_test1.jsonl")) # .pkl


    def process_pkldata(self,filepaths):
        print("filepath size:",len(filepaths))
        for filepath in filepaths:
            print("Processing:",filepath)
            self.split_and_save_content(filepath, Args.train_ratio, Args.dev_ratio, Args.data_pkl_path)


def data_generator(file_paths, batch_size, files_per_batch=10):
    batch_data = []
    for i in range(0, len(file_paths), files_per_batch):
        current_files = file_paths[i:i + files_per_batch]
        for file_path in current_files:
            with open(file_path, 'r') as file:
                for line in file:
                    # 对每一行进行预处理，例如解析JSON，清洗文本等
                    processed_data = preprocess(line)
                    
                    batch_data.append(processed_data)
                    if len(batch_data) == batch_size:
                        yield batch_data
                        batch_data = []
        # 确保最后的数据也被处理
        if batch_data:
            yield batch_data
            batch_data = []

# 使用示例
file_paths = ['file1.txt', 'file2.txt', ...]  # 你的文件路径列表
batch_size = 128  # 选择一个合适的批次大小

for batch in data_generator(file_paths, batch_size):
    # 这里是你的训练代码，例如模型训练
    train_model(batch)

