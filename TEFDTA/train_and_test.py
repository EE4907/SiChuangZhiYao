import torch
import copy
import logging
import os
from metrics import get_cindex, get_rm2

# 创建保存日志目录
os.makedirs("TEFDTA/model", exist_ok=True)
os.makedirs("TEFDTA/logs", exist_ok=True)

# 自定义CSV日志器
class SimpleCSVLogger:
    def __init__(self, save_dir, name):
        self.save_path = os.path.join(save_dir, f"{name}_metrics.csv")
        os.makedirs(save_dir, exist_ok=True)
        with open(self.save_path, 'w', newline='') as f:
            f.write("epoch,train_loss,val_loss,val_accuracy\n")

    def log_metrics(self, metrics, epoch):
        with open(self.save_path, 'a', newline='') as f:
            f.write(f"{epoch},{metrics.get('train_loss','')},{metrics.get('val_loss','')},{metrics.get('val_accuracy','')}\n")


# 测试集评估函数
def test(data_loader, model, loss_fn):
    with torch.no_grad():
        model.eval()
        y_true = []
        y_pred = []
        score_list = []
        label_list = []
        smiles_list = []
        fasta_list = []

        running_loss = 0.0
        for sample in data_loader:
            smiles, fasta, label = sample
            score = model(smiles, fasta).view(-1)
            loss = loss_fn(score, label)
            running_loss += loss.item()
            y_pred += score.detach().cpu().tolist()
            y_true += label.detach().cpu().tolist()
            score_list.append(score)
            label_list.append(label)
            smiles_list.append(smiles)
            fasta_list.append(fasta)

        # 保存预测结果（覆盖写）
        with open("prediction.txt", 'w') as f:
            for i in range(len(score_list)):
                f.write(f"{smiles_list[i]} {fasta_list[i]} {label_list[i]} {score_list[i]}\n")

        ci = get_cindex(y_true, y_pred)
        rm2 = get_rm2(y_true, y_pred)
        model.train()
    return running_loss / len(data_loader), ci, rm2


# 主训练函数
def train(model, train_loader, val_loader, test_loader, writer, NAME, lr=0.0001, epoch=1):
    device = torch.device("cpu")
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model_best = copy.deepcopy(model)
    min_loss = float('inf')

    for epo in range(epoch):
        print(f"Epoch {epo+1}/{epoch}")
        model.train()
        running_loss = 0.0

        print("len of train_loader:", len(train_loader))
        for data in train_loader:
            smiles, fasta, label = data
            
            # print("forward")
            score = model(smiles, fasta).view(-1).to(device)
            loss = loss_fn(score, label)
            
            opt.zero_grad()
            # print("backward")
            loss.backward()
            opt.step()

            running_loss += loss.item()

        # 在验证集上测试
        val_loss, val_ci, val_rm2 = test(val_loader, model, loss_fn)

        # 日志记录
        writer.log_metrics({
            'train_loss': running_loss / len(train_loader),
            'val_loss': val_loss,
            'val_accuracy': val_ci
        }, epoch=epo)

        logging.info(
            f"Epoch {epo+1}: train_loss={running_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}, ci={val_ci:.4f}, rm2={val_rm2:.4f}"
        )

        # 保存最好的模型
        if val_loss < min_loss:
            min_loss = val_loss
            model_best = copy.deepcopy(model)

    # 在测试集上测试并保存最终模型
    test_loss, test_ci, test_rm2 = test(test_loader, model_best, loss_fn)
    logging.info(f"Test loss: {test_loss:.4f}, ci={test_ci:.4f}, rm2={test_rm2:.4f}")
    torch.save(model_best, f"TEFDTA/model/{NAME}.pth")

