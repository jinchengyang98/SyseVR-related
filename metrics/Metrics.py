import torchmetrics
from config.Trainingconfig import TrainingConfig as Args
# 类实例化
Args = Args()
class PerformanceMetrics:
    def __init__(self, num_classes = Args.num_classes,device = Args.device):
        # Accuracy: 计算分类正确的样本比例
        self.accuracy_metric = torchmetrics.Accuracy(task=Args.task_type, num_classes=num_classes).to(device) # task="multiclass", num_classes=3, top_k=2

        # Precision: 计算每个类的精确度（预测为正且实际为正的比例），然后取平均值。
        # "Macro" 平均意味着每个类别的指标被等同重视，无论它们的样本数目多少。
        self.precision_metric = torchmetrics.Precision(task=Args.task_type,num_classes=num_classes, average='macro').to(device)

        # Recall: 计算每个类的召回率（实际为正且预测为正的比例），然后取平均值。
        # "Macro" 平均意味着每个类别的指标被等同重视，无论它们的样本数目多少。
        self.recall_metric = torchmetrics.Recall(task=Args.task_type,num_classes=num_classes, average='macro').to(device)

        # F1 Score: 计算 F1 分数，它是精确度和召回率的调和平均值。
        # "Macro" 平均意味着每个类别的指标被等同重视，无论它们的样本数目多少。
        # self.f1_metric = torchmetrics.F1(task=Args.task_type,num_classes=num_classes, average='macro') # tochmetrics.F1()无函数

        # Confusion Matrix: 计算混淆矩阵，它是一个表格，用于描述分类模型的性能，
        # 显示了每个类别被正确和错误分类的次数。
        self.confmat_metric = torchmetrics.ConfusionMatrix(task=Args.task_type,num_classes=num_classes).to(device)

        # AUC-ROC: 计算接收者操作特征曲线（ROC）下的面积（AUC）。
        # 这是一个用于二分类问题的性能指标，可以通过将问题视为一对多问题来用于多类分类。
        self.auc_metric = torchmetrics.AUROC(task=Args.task_type,num_classes=num_classes).to(device)


    def update_metrics(self, preds, targets):
        self.accuracy_metric.update(preds, targets)
        self.precision_metric.update(preds, targets)
        self.recall_metric.update(preds, targets)
        # self.f1_metric.update(preds, targets)
        self.confmat_metric.update(preds, targets)
        self.auc_metric.update(preds, targets)

    def compute_metrics(self):
        accuracy = self.accuracy_metric.compute()
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        # f1_score = self.f1_metric.compute()
        confmat = self.confmat_metric.compute()
        auc = self.auc_metric.compute()
        return accuracy, precision, recall, confmat, auc

    def reset_metrics(self):
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        # self.f1_metric.reset()
        self.confmat_metric.reset()
        self.auc_metric.reset()
