import csv,os
from config import Trainingconfig as Args
Args = Args.TrainingConfig()
class TrainingMetricsLogger:
    def __init__(self, save_path = Args.run_save_path , mode = "train"):
        self.mode = mode
        self.filename = save_path + Args.task_name # + "_"+ self.mode +"_metrics.csv"
        if os.path.exists(self.filename)!= True:
            os.makedirs(self.filename)
            
        self.file = open(self.filename + "_"+ self.mode +"_metrics.csv" , mode='w', newline='', encoding='utf-8')
        self.writer = None
        self.headers_written = False

    def log(self, metrics):
        # 记录每个epoch的数据
        # 如果标题行还未写入，则在第一次记录数据时写入
        if not self.headers_written:
            self.writer = csv.DictWriter(self.file, fieldnames=metrics.keys())
            self.writer.writeheader()
            self.headers_written = True

        self.writer.writerow(metrics)

    def close(self):
        # 关闭文件
        self.file.close()

# # 使用示例
# logger = TrainingMetricsLogger('training_metrics.csv')

# total_epochs = 10  # 假设的训练周期总数
# for epoch in range(total_epochs):
#     # 进行训练...
#     # 假设计算了loss和accuracy
#     loss = epoch * 0.1  # 示例数据
#     accuracy = epoch * 0.05  # 示例数据
#     logger.log(epoch, loss, accuracy)

# logger.close()
