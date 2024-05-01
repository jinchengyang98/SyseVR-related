
import torch
from torch import nn as nn
import model.siamese_network as Siamese_network
from dataprocess.Dataset import SiaMese_Dataset
#超参数
from config.Trainingconfig import TrainingConfig as Args
from metrics.Metrics import PerformanceMetrics as Metrics
# from torch.utils.data import DataLoader
import os
from dataprocess.Dataloader import DataLoader as DataLoader
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(1)
## 类实例化
Args = Args() # 
Metrics = Metrics()
# 加载词表
from ncc.data.dictionary import TransformersDictionary
# #bpe dictionary
pretrain_path = os.path.join(Args.root_data_path, "pretrain/")
vocab = TransformersDictionary.from_pretrained(pretrain_path + "microsoft/codebert-base", do_lower_case=False) # arg1：预训练模型的名称，arg2：是否小写转换
# 加载画图函数

from draw.draw import TrainingMetricsLogger as Draw

# 定义训练循环
def train(model, train_dataset, optimizer, criterion, epoch, device):
    draw_logger = Draw(mode = "train") # 记录一些指标
    model.train()
    total_loss = 0
    total_batch = 0
    for idx in range(5):
        batch = next(train_dataset)
        x1, x2, targets = batch['x1'].to(device), batch['x2'].to(device), batch['target'].to(device) # to(device)
        # 将tensor转移到指定的设备上（GPU或CPU）
        targets = targets.squeeze(1)

        optimizer.zero_grad() # 梯度清零
        outputs = model(x1, x2).to(device) # 此处根据模型的输入计算输出
        # print("outputs",type(outputs))
        # print("outputs",type(targets))
        # targets = targets.float()
        print(outputs.size())
        print(targets.size())
        loss= criterion(outputs.to(device), targets.to(device)) # 此处根据模型的输出和标签计算损失

        loss.backward() # 反向传播
        optimizer.step()

        # 累计loss
        total_loss += loss.item()

        # 更新指标
        preds = torch.argmax(outputs, dim=-1).to(device)
        # print("targets:",targets)
        # print("output:",outputs)
        Metrics.update_metrics(preds = preds.to(device), targets = targets.to(device))
        total_batch = total_batch + len(batch)
        print(len(batch))
    # 计算指标
    # print("total batch:",total_batch)
    accuracy, precision, recall , confmat, auc = Metrics.compute_metrics()
    avg_loss = total_loss / len(train_dataset)
    print(f'Train Metrics: Loss: {avg_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall} , AUC: {auc}')
    metrics = {
        "epoch":epoch,
        "loss" : avg_loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc
    }
    draw_logger.log(metrics)
    Metrics.reset_metrics()


def evaluate(model, dev_dataset, device,mode='eval'):
    draw_logger = Draw(mode = "eval") # 记录一些指标
    model.eval()
    total_acc, total_count,total_loss = 0,0 ,0

    with torch.no_grad():
        for idx in range(1):
            batch = next(dev_dataset)
            x1, x2, targets = batch['x1'].to(device), batch['x2'].to(device), batch['target'].to(device)
            targets = targets.squeeze(1)
            outputs = model(x1, x2)
            loss = criterion(outputs, targets) # 此处根据模型的输出和标签计算损失

            # 累计loss
            total_loss += loss.item()

            # 更新指标
            preds = torch.argmax(outputs, dim=1) # 按行取最大值
            Metrics.update_metrics(preds, targets)

        # 计算指标
        accuracy, precision, recall, confmat, auc = Metrics.compute_metrics()
        avg_loss = total_loss / len(dev_dataset)
        print(mode,f'Metrics: Loss: {avg_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall} , AUC: {auc}')
        metrics = {
            "epoch":epoch,
            "loss" : avg_loss,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "AUC": auc
        }
        draw_logger.log(metrics)
        Metrics.reset_metrics()


if __name__ == "__main__":
    # 1. 定义数据加载器

    Dataloader = DataLoader(dictionary = vocab)
    # 2. 加载数据集
    for mode in Args.task_mode:
        Dataloader.load_dataset(split=mode, data_file = Args.data_pkl_path + Args.task_name + "_mul_" + mode ) # + Args.task_name + "_" + mode
        

    dataset = {}
    for mode in Args.task_mode:
        dataset[mode] = Dataloader.get_batch_iterator(dataset=Dataloader.dataset(mode), max_sentences=Args.batch_size).\
                        next_epoch_itr(shuffle=True) #FIXME

    # 3. 定义模型

    model = Siamese_network.Demo_SiameseLSTM.build_model(dictionary=vocab)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss() # torch实现会自动将梯度转换为loss
    optimizer = torch.optim.Adam(model.parameters(), lr=Args.learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # device = "cpu"
    print("device:",device)

    # 5. 开始训练

    for epoch in range(Args.num_epochs):
        train(model, dataset["train"], optimizer, criterion, epoch, device) # train 
        # 评估模型
        evaluate(model, dataset["eval"], device,mode='eval')

    # 6. 保存模型
    torch.save(model.state_dict(), Args.save_path)
    # 7. 测试模型
    model.load_state_dict(torch.load(Args.save_path)) # torchvision==0.14.0 torchaudio==0.13.0 
    model.to(device) # 将模型加载到指定的设备上（GPU或CPU）
    evaluate(model, Dataloader["test"], device,mode='Test')

