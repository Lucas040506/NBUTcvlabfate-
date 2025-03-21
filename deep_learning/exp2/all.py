import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

# 读取数据并进行预处理
def read_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                word_label_pairs = line.split()
                words = []
                labs = []
                for pair in word_label_pairs:
                    try:
                        word, label = pair.rsplit('/', 1)
                        words.append(word)
                        labs.append(label)
                    except ValueError:
                        print(f"Invalid pair format: {pair} in line: {line}. Skipping this pair.")
                if words:
                    texts.append(words)
                    labels.append(labs)
    return texts, labels

# 构建词汇表和标签表
def build_vocab(texts, labels):
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    label2idx = {'O': 0}
    for text in texts:
        for word in text:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    for lab_list in labels:
        for lab in lab_list:
            if lab not in label2idx:
                label2idx[lab] = len(label2idx)
    return word2idx, label2idx

# 将文本和标签转换为索引序列
def text_to_idx(texts, labels, word2idx, label2idx):
    text_idx = []
    label_idx = []
    for text, lab in zip(texts, labels):
        text_idx.append([word2idx.get(word, word2idx['<UNK>']) for word in text])
        label_idx.append([label2idx[lab] for lab in lab])
    return text_idx, label_idx

# 自定义数据集类
class NERDataset(Dataset):
    def __init__(self, text_idx, label_idx):
        self.text_idx = text_idx
        self.label_idx = label_idx

    def __len__(self):
        return len(self.text_idx)

    def __getitem__(self, idx):
        return torch.tensor(self.text_idx[idx]), torch.tensor(self.label_idx[idx])

# 自定义 collate_fn 函数
def custom_collate(batch):
    texts, labels = zip(*batch)
    max_length = max(len(text) for text in texts)
    padded_texts = []
    padded_labels = []
    for text, label in zip(texts, labels):
        padding_length = max_length - len(text)
        padded_text = torch.cat([text, torch.zeros(padding_length, dtype=text.dtype)])
        padded_label = torch.cat([label, torch.zeros(padding_length, dtype=label.dtype)])
        padded_texts.append(padded_text)
        padded_labels.append(padded_label)
    return torch.stack(padded_texts), torch.stack(padded_labels)

# RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

# GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

# BiGRU模型
class BiGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BiGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.bigru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.bigru(x, h0)
        out = self.fc(out)
        return out

# BiLSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.bilstm(x, (h0, c0))
        out = self.fc(out)
        return out

# 训练模型
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_tokens = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs.view(-1, outputs.size(2)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_tokens += texts.numel()  # 计算处理的tokens数量
    return loss.item(), total_tokens

# 评估模型
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

# 更新evaluate_model函数，添加混淆矩阵
def evaluate_model(model, test_loader, criterion, device, label2idx):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs.view(-1, outputs.size(2)), labels.view(-1))
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 2)
            total += labels.numel()
            correct += (predicted == labels).sum().item()

            # 过滤掉'O'标签
            mask = (labels != label2idx['O'])
            all_labels.extend(labels[mask].cpu().numpy())
            all_predictions.extend(predicted[mask].cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print(f"Confusion Matrix (without 'O' label):")
    # 使用seaborn绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(label2idx.keys())[1:], yticklabels=list(label2idx.keys())[1:])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    return total_loss / len(test_loader), accuracy, precision, recall, f1

# 在主函数中调用
def main():
    file_path = r'命名实体识别标记语料.txt'
    texts, labels = read_data(file_path)
    word2idx, label2idx = build_vocab(texts, labels)
    text_idx, label_idx = text_to_idx(texts, labels, word2idx, label2idx)

    input_dim = len(word2idx)
    output_dim = len(label2idx)
    hidden_dim = 128
    num_layers = 2
    batch_size = 32
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 划分数据集为训练集、验证集和测试集
    train_texts = text_idx[:int(len(text_idx) * 0.7)]
    train_labels = label_idx[:int(len(label_idx) * 0.7)]
    val_texts = text_idx[int(len(text_idx) * 0.7):int(len(text_idx) * 0.85)]
    val_labels = label_idx[int(len(label_idx) * 0.7):int(len(label_idx) * 0.85)]
    test_texts = text_idx[int(len(text_idx) * 0.85):]
    test_labels = label_idx[int(len(label_idx) * 0.85):]

    train_dataset = NERDataset(train_texts, train_labels)
    val_dataset = NERDataset(val_texts, val_labels)
    test_dataset = NERDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    models = {
        'RNN': RNNModel(input_dim, hidden_dim, output_dim, num_layers),
        'GRU': GRUModel(input_dim, hidden_dim, output_dim, num_layers),
        'LSTM': LSTMModel(input_dim, hidden_dim, output_dim, num_layers),
        'BiGRU': BiGRUModel(input_dim, hidden_dim, output_dim, num_layers),
        'BiLSTM': BiLSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    }

    criterion = nn.CrossEntropyLoss()
    results = {}
    train_losses = {name: [] for name in models.keys()}
    val_losses = {name: [] for name in models.keys()}
    val_accuracies = {name: [] for name in models.keys()}
    train_speeds = {name: [] for name in models.keys()}

    for model_name, model in models.items():
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_val_loss = float('inf')
        for epoch in range(epochs):
            epoch_loss, total_tokens = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device, label2idx)
            train_losses[model_name].append(epoch_loss)
            val_losses[model_name].append(val_loss)
            val_accuracies[model_name].append(val_accuracy)
            train_speed = total_tokens / (len(train_loader) * batch_size)  # 计算tokens/s
            train_speeds[model_name].append(train_speed)

            print(f"{model_name} Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Speed: {train_speed:.2f} tokens/s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()

        test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, device, label2idx)
        results[model_name] = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
        print(f"{model_name} Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1: {test_f1}")

    # 绘制训练损失对比图
    plt.figure(figsize=(12, 6))
    for model_name, losses in train_losses.items():
        plt.plot(losses, label=model_name)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制验证损失对比图
    plt.figure(figsize=(12, 6))
    for model_name, losses in val_losses.items():
        plt.plot(losses, label=model_name)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制验证准确率对比图
    plt.figure(figsize=(12, 6))
    for model_name, accuracies in val_accuracies.items():
        plt.plot(accuracies, label=model_name)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # 绘制训练速度对比图
    plt.figure(figsize=(12, 6))
    for model_name, speeds in train_speeds.items():
        plt.plot(speeds, label=model_name)
    plt.title('Training Speed Comparison (tokens/s)')
    plt.xlabel('Epoch')
    plt.ylabel('Tokens/s')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

