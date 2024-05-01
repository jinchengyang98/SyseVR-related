
import torch
import torch.nn as nn
import model.siamese_network as Siamese_network 
from dataprocess.Data_proc import Siamese_Dataset, TextPairTestDataset
#超参数
from config.Trainingconfig import TrainingConfig as Args
from metrics.Metrics import PerformanceMetrics as Metrics
from torch.utils.data import DataLoader
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(1)
## 类实例化
Args = Args() # 类实例化
Metrics = Metrics()

# 定义训练循环
def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    total_loss = 0
    total_batch = 0
    for batch in train_loader:
        x1, x2, targets = batch['x1'].to(device), batch['x2'].to(device), batch['y'].to(device)

        optimizer.zero_grad() # 梯度清零
        outputs = model(x1, x2).to(device) # 此处根据模型的输入计算输出
        # print("outputs",type(outputs))
        # print("outputs",type(targets))
        # targets = targets.float()
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
    avg_loss = total_loss / len(train_loader)
    print(f'Train Metrics: Loss: {avg_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall} , AUC: {auc}')

    Metrics.reset_metrics()


def evaluate(model, dev_loader, device,mode='dev'):
    model.eval()
    total_acc, total_count,total_loss = 0,0 ,0

    with torch.no_grad():
        for batch in dev_loader:
            x1, x2, targets = batch['x1'].to(device), batch['x2'].to(device), batch['y'].to(device)

            outputs = model(x1, x2)
            loss = criterion(outputs, targets) # 此处根据模型的输出和标签计算损失

            # 累计loss
            total_loss += loss.item()

            # 更新指标
            preds = torch.argmax(outputs, dim=1) # 按行取最大值
            Metrics.update_metrics(preds, targets)

        # 计算指标
        accuracy, precision, recall, confmat, auc = Metrics.compute_metrics()
        avg_loss = total_loss / len(dev_loader)
        print(mode,f'Metrics: Loss: {avg_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall} , AUC: {auc}')

        Metrics.reset_metrics()


if __name__ == "__main__":


    # 1. 加载数据集
    dataset , Dataloader = {},{}
    for mode in Args.task_mode:
        dataset[mode] = Siamese_Dataset(Args.data_pkl_path + Args.task_name + "_" + mode + ".json", Args.max_seq_length)
        print("数据集大小" + mode , len(dataset[mode].indexes))
    # train_dataset = Siamese_Dataset(Args.data_pkl_path + Args.task_name + "_train", Args.max_seq_length)
    # dev_dataset = Siamese_Dataset(Args.data_pkl_path + Args.task_name + "_eval")
    # test_dataset = Siamese_Dataset(Args.data_pkl_path + Args.task_name + "_test")

    # 2. 创建dataloader
    for mode in Args.task_mode:
        Dataloader[mode] = DataLoader(batch_size=Args.batch_size,num_workers=Args.num_load_worker,shuffle=Args.is_shuffle,dataset=dataset[mode])

    # train_loader = DataLoader.create_data_loaders(train_dataset, batch_size=Args.batch_size)
    # dev_loader = DataLoader.create_data_loaders(dev_dataset, batch_size=Args.batch_size)
    # test_loader = DataLoader.create_data_loaders(test_dataset, batch_size=Args.batch_size)

    # 3. 定义模型

    model = Siamese_network.SiameseLSTM(batch_size = Args.batch_size, hidden_units= Args.lstm_hidden_units
                                        , vocab_size = dataset["train"].vocab.num_words, embedding_size= Args.embedding_dim,num_layers=Args.lstm_num_layers)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Args.learning_rate)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    print("device:",device)
    # device = "cpu"
    # 5. 开始训练

    model.to(device)
    for epoch in range(Args.num_epochs):
        train(model, Dataloader["train"], optimizer, criterion, epoch, device)
        # 评估模型
        evaluate(model, Dataloader["eval"], device,mode='Dev')

    # 5. 保存模型
    torch.save(model.state_dict(), Args.save_path)
    # 6. 测试模型
    model.load_state_dict(torch.load(Args.save_path)) # torchvision==0.14.0 torchaudio==0.13.0 
    model.to(device) # 将模型加载到指定的设备上（GPU或CPU）
    evaluate(model, Dataloader["test"], device,mode='Test')

