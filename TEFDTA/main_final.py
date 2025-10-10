import torch
import logging
import pandas as pd
import argparse
import os
from torch.utils.data import DataLoader, random_split

from model import Classifier, SMILESModel, FASTAModel
from process_data import DTAData, CHARISOSMILEN, CHARPROTLEN, MACCSLEN
from train_and_test import train

# 简化CSV日志
import csv
class SimpleCSVLogger:
    def __init__(self, save_dir, name):
        self.save_path = os.path.join(save_dir, f"{name}_metrics.csv")
        os.makedirs(save_dir, exist_ok=True)
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy'])
    def log_metrics(self, metrics, epoch):
        with open(self.save_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                metrics.get('train_loss', ''),
                metrics.get('val_loss', ''),
                metrics.get('val_accuracy', '')
            ])

MODEL_NAME = "FEDTA"
BATCH_SIZE = 256
DATASET = "davis"  # 保持原来逻辑

logging.basicConfig(filename=f'TEFDTA/{MODEL_NAME}.log', level=logging.DEBUG)
device = torch.device("cpu")

def main(args):
    if args.mode == 'predict':
        # 1. 读取输入
        smiles_list = [l.strip() for l in open(args.smiles_file) if l.strip()]

        smiles_list = [smile.split()[0] for smile in smiles_list]
        # print(len(smiles_list), "SMILES loaded.")
        fasta = ''.join([l.strip() for l in open(args.protein_fasta) if not l.startswith('>')])
        # print("FASTA loaded, length:", len(fasta))
        fasta_list = [fasta] * len(smiles_list)
        label_list = [0] * len(smiles_list)
        # 2. 构建数据集
        max_smiles_len = 100
        max_fasta_len = 1000
        dataset = DTAData(smiles_list, fasta_list, label_list, device, max_smiles_len, max_fasta_len)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        # 3. 加载模型结构与权重
        smiles_model = SMILESModel(char_set_len=MACCSLEN)
        fasta_model = FASTAModel(char_set_len=CHARPROTLEN+1)
        model = Classifier(smiles_model, fasta_model)
        # model_path = f"weights/{args.name}.pth"
        model_path = f"TEFDTA/model/mymodel.pth"
        
        #model = model.to(device).eval()
        model = torch.load(model_path, map_location=device)
        model.eval()
        # 4. 推理

        print("Starting prediction...")
        preds = []
        with torch.no_grad():
            print(len(loader))
            for x, y, _ in loader:
                print(x.shape, y.shape)
                print(x[0], y[0])
                preds += model(x.to(device), y.to(device)).squeeze().cpu().tolist()
        # 5. 保存结果
        outdir = f"TEFDTA/result/{args.name}"
        os.makedirs(outdir, exist_ok=True)
        pd.DataFrame({'smiles': smiles_list, 'score': preds}).to_csv(f"{outdir}/prediction.csv", index=False)
        print(f"✅ Prediction done. Saved to {outdir}/prediction.csv")
        return

    # 以下为训练流程
    csv_logger = SimpleCSVLogger("logs", args.name)
    if DATASET == 'davis':
        df_train = pd.read_csv('TEFDTA/data/Davis/Davis_train.csv')
        df_test = pd.read_csv('TEFDTA/data/Davis/Davis_test.csv')
        max_smiles_len, max_fasta_len = 85, 1000
    elif DATASET == 'kiba':
        df_train = pd.read_csv('TEFDTA/data/KIBA/KIBA_train.csv')
        df_test = pd.read_csv('TEFDTA/data/KIBA/KIBA_test.csv')
        max_smiles_len, max_fasta_len = 100, 1000
    else:  # 'Bind'
        df_train = pd.read_csv('TEFDTA/data/BindingDB/BindingDB_train.csv')
        df_test = pd.read_csv('TEFDTA/data/BindingDB/BindingDB_test.csv')
        max_smiles_len, max_fasta_len = 100, 1000

    train_set = DTAData(list(df_train.iso_smiles), list(df_train.target_sequence),
                        list(df_train.affinity), device, max_smiles_len, max_fasta_len)
    test_set = DTAData(list(df_test.iso_smiles), list(df_test.target_sequence),
                       list(df_test.affinity), device, max_smiles_len, max_fasta_len)
    t_size = int(len(train_set)*0.8)
    train_set, val_set = random_split(train_set, [t_size, len(train_set)-t_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 构造模型并启动训练
    smiles_model = SMILESModel(char_set_len=MACCSLEN)
    fasta_model = FASTAModel(char_set_len=CHARPROTLEN+1)
    model = Classifier(smiles_model, fasta_model).to(device)
    train(model, train_loader, val_loader, test_loader,
          writer=csv_logger, NAME=args.name, lr=0.0001, epoch=1)
    # 训练完后会自动保存权重到 weights/{args.name}.pth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--smiles_file', type=str, help='predict mode: input SMILES .txt')
    parser.add_argument('--protein_fasta', type=str, help='predict mode: input FASTA file')
    parser.add_argument('--name', type=str, required=True, help='model name for save/predict')
    args = parser.parse_args()
    main(args)

   