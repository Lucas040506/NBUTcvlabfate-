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


class InceptionModule(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(InceptionModule, self).__init__()

        # 1x1 卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 卷积 -> 3x3 卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1 卷积 -> 5x5 卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3 最大池化 -> 1x1 卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 沿着通道维度拼接各分支的输出
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 5x5 平均池化，步长为3
        x = self.avgpool(x)
        # 1x1 卷积
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # 展平
        x = torch.flatten(x, 1)
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        # 第一阶段
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # 第二阶段
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        # Inception模块
        # 第三阶段
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)  # 输出通道: 256
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)  # 输出通道: 480
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        # 第四阶段
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)  # 输出通道: 512
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)  # 输出通道: 512
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)  # 输出通道: 512
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)  # 输出通道: 528
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)  # 输出通道: 832
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        # 第五阶段
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)  # 输出通道: 832
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)  # 输出通道: 1024

        # 辅助分类器
        if aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes)
            self.aux2 = AuxiliaryClassifier(528, num_classes)

        # 平均池化和分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        # 初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 第一阶段
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        # 第二阶段
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool2(x)

        # 第三阶段
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # 第四阶段
        x = self.inception4a(x)

        # 辅助分类器1
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # 辅助分类器2
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        # 第五阶段
        x = self.inception5a(x)
        x = self.inception5b(x)

        # 平均池化和分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# 创建GoogLeNet模型的便捷函数
def googlenet(num_classes=1000, aux_logits=True, pretrained=False):
    model = GoogLeNet(num_classes=num_classes, aux_logits=aux_logits)
    return model


def filter_specific_classes(dataset, selected_classes):
    """
    Filter dataset to include only the specified classes

    Args:
        dataset: The original dataset
        selected_classes: List of class indices to keep

    Returns:
        List of filtered samples
    """
    filtered_data = [sample for sample in dataset.samples if sample[1] in selected_classes]
    return filtered_data


def remap_class_indices(samples, class_mapping):
    """
    Remap the class indices in the dataset samples

    Args:
        samples: Dataset samples
        class_mapping: Dictionary mapping original indices to new indices

    Returns:
        List of samples with remapped class indices
    """
    return [(sample[0], class_mapping[sample[1]]) for sample in samples]


def evaluate_model(model, val_loader, device, class_names):
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

    # Output classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    return all_preds, all_labels


if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")

    # Specify the classes we want to keep (1, 5, 7)
    # abies_concolor acer_griseum aesculus_flava ailanthus_altissima amelanchier_arborea betula_alleghaniensis carpinus_betulus cedrus_deodara cladrastis_lutea diospyros_virginiana
    selected_classes = [1, 5, 14, 18, 20, 24, 30, 40, 50, 60 ]
    num_classes = len(selected_classes)

    # Create a mapping from original class indices to new consecutive indices
    class_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(selected_classes)}

    #model = SimpleAlexNet(num_classes=num_classes).to(device)
    model = GoogLeNet(num_classes=10, aux_logits=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_root = "../dataset"
    batch_size = 16

    # Load the original datasets
    train_dataset = datasets.ImageFolder(root=f"{data_root}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_root}/val", transform=transform)

    # Get the original class names
    original_class_names = train_dataset.classes
    selected_class_names = [original_class_names[i] for i in selected_classes]
    print(f"Selected classes: {selected_class_names}")

    # Filter datasets to include only classes 1, 5, and 7
    train_dataset.samples = filter_specific_classes(train_dataset, selected_classes)
    val_dataset.samples = filter_specific_classes(val_dataset, selected_classes)

    # Remap class indices to be consecutive (0, 1, 2)
    train_dataset.samples = remap_class_indices(train_dataset.samples, class_mapping)
    val_dataset.samples = remap_class_indices(val_dataset.samples, class_mapping)

    # Update class_to_idx mapping
    train_dataset.class_to_idx = {original_class_names[class_idx]: class_mapping[class_idx]
                                  for class_idx in selected_classes}
    val_dataset.class_to_idx = {original_class_names[class_idx]: class_mapping[class_idx]
                                for class_idx in selected_classes}

    # Updated class names (ordered by new indices)
    class_names = [original_class_names[idx] for idx in selected_classes]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training with {num_classes} classes: {class_names}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_acc_history, val_acc_history, train_loss_history, val_loss_history = [], [], [], []
    best_val_acc = 0
    patience = 10
    wait = 0

    # 修改训练循环中的损失计算部分
    for epoch in range(100):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # 处理模型输出
            outputs = model(inputs)

            # 处理辅助分类器输出
            if isinstance(outputs, tuple):
                # 计算综合损失：主损失 + 辅助损失 * 权重
                loss = criterion(outputs[0], labels) + 0.3 * criterion(outputs[1], labels) + 0.3 * criterion(outputs[2],
                                                                                                             labels)
                outputs = outputs[0]  # 用主分类器输出计算准确率
            else:
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

                # 在评估模式下，模型只返回主分类器输出，但为安全起见，添加检查
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

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
        print(
            f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered. Evaluating model...")
                all_preds, all_labels = evaluate_model(model, val_loader, device, class_names)
                break

    # Plot accuracy and loss curves
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

    # Generate confusion matrix after training is complete
    all_preds, all_labels = evaluate_model(model, val_loader, device, class_names)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()

