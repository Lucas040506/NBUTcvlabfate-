import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self._calculate_flatten_size()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.flatten_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def _calculate_flatten_size(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            x = self.features(x)
            self.flatten_size = x.numel() // x.shape[0]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class NIN(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN, self).__init__()
        # First block - initial feature extraction
        self.block1 = nn.Sequential(
            nin_block(3, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.2)
        )
        # Second block - mid-level features
        self.block2 = nn.Sequential(
            nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.3)
        )
        # Third block - higher-level features
        self.block3 = nn.Sequential(
            nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.4)
        )
        # Final block - classification
        self.block4 = nn.Sequential(
            nin_block(384, num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


def filter_dataset(dataset, num_classes):
    # 创建一个新的数据集，只包含前num_classes类
    filtered_data = [sample for sample in dataset.samples if sample[1] < num_classes]
    return filtered_data


def evaluate_model(model, val_loader, device, val_dataset):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 获取目标类名
    target_names = [val_dataset.classes[i] for i in range(len(val_dataset.classes)) if i < 10]

    # 输出分类报告
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\nClassification Report:")
    print(report)

if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")

    model = NIN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_root = "../dataset"
    batch_size = 64

    # 加载原始数据集
    train_dataset = datasets.ImageFolder(root=f"{data_root}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_root}/val", transform=transform)

    # 过滤数据集，只保留前10类
    train_dataset.samples = filter_dataset(train_dataset, 10)
    val_dataset.samples = filter_dataset(val_dataset, 10)

    # 更新class_to_idx映射
    train_dataset.class_to_idx = {cls: idx for cls, idx in train_dataset.class_to_idx.items() if idx < 10}
    val_dataset.class_to_idx = {cls: idx for cls, idx in val_dataset.class_to_idx.items() if idx < 10}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = 10
    print(f"数据集中包含 {num_classes} 个类别")

    train_acc_history, val_acc_history, train_loss_history, val_loss_history = [], [], [], []
    best_val_acc = 0
    patience = 10
    wait = 0

    for epoch in range(100):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)

        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= total
        val_acc = correct / total
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)

        scheduler.step(val_loss)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered. Evaluating model...")
                evaluate_model(model, val_loader, device, val_dataset)  # 进行评估
                break  # 可以在这里选择是否停止训练

    plt.figure()
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.savefig("accuracy_curve.png")
    plt.show()

    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig("loss_curve.png")
    plt.show()

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()
