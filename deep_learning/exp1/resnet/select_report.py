import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch.nn.functional as F


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # 如果输入输出尺寸不匹配，需要使用1x1卷积进行调整
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 添加残差连接
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    # 瓶颈结构，输出通道数是输入的4倍
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 全局平均池化和全连接分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def feature_extract(self, x):
        """返回特征向量，用于迁移学习或特征可视化"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        return features


# 定义不同大小的ResNet模型
def ResNet18(num_classes=10, input_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channels)


def ResNet34(num_classes=10, input_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_channels)


def ResNet50(num_classes=10, input_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_channels)


def ResNet101(num_classes=10, input_channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, input_channels)


def ResNet152(num_classes=10, input_channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, input_channels)


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

    model = ResNet18(num_classes=num_classes).to(device)
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
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

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

