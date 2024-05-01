import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from model import bgru as BiGRUNetwork  # 请确保此处路径和名称正确
from dataprocess import Dataset  # 数据加载类，自行定义

from config.Trainingconfig import TrainingConfig  # 训练配置
from metrics import Metrics  # 性能评估指标
config = TrainingConfig()
# 设定环境变量和设备
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(1)  # 指定GPU设备
from Log import Log  # 日志配置
# Set up logging
Log.setup_logging()
import logging  #
# 初始化配置和性能评估工具
metrics = Metrics.PerformanceMetrics()

# 定义训练函数
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    metrics.reset_metrics()
    total_loss = 0

    for batch in train_loader:
        # 如果批次数据是元组形式
        batch = tuple(t.to(device) for t in batch)
        data, targets = batch  # 这里假设每个批次返回的是 (data, targets) 形式的元组

        # 然后单独转换
        # data = data.to(device)
        # targets = targets.to(device)

        # data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        metrics.update_metrics(preds=preds, targets=targets)

    avg_loss = total_loss / len(train_loader)
    accuracy, precision, recall, confmat, auc = metrics.compute_metrics()
    logging.info(f'Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}')
    logging.info(f'Confusion Matrix:\n{confmat}')

# 定义评估函数
def evaluate(model, dev_loader, criterion, device,mode='Validation'):
    model.eval()
    metrics.reset_metrics()
    total_loss = 0

    with torch.no_grad():
        for data, targets in dev_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            metrics.update_metrics(preds=preds, targets=targets)

    avg_loss = total_loss / len(dev_loader)
    accuracy, precision, recall, confmat, auc = metrics.compute_metrics()
    logging.info(f'{mode} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}')
    logging.info(f'Confusion Matrix:\n{confmat}')

# 主函数
if __name__ == "__main__":
    # 加载数据集
    Vul_dataset = Dataset.TextDataset(config.slice_data_path)
    train_dataset, dev_dataset, test_dataset = Vul_dataset.split_dataset()

    # data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    # 初始化模型
    model = BiGRUNetwork.BiGRUNetwork(vocab_size=Vul_dataset.voc_size, embedding_dim=config.BGRU_embedding_dim,
                         hidden_units=config.BGRU_hidden_units, num_layers=config.BGRU_num_layers, num_classes=config.BGRU_num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 开始训练和评估
    for epoch in range(config.num_epochs):
        print(f'Starting epoch {epoch+1}')
        train(model, train_loader, optimizer, criterion, device)
        evaluate(model, dev_loader, criterion, device)

    # 可以在这里保存模型
    torch.save(model.state_dict(), config.model_save_path)
