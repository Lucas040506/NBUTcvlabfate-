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

class ImprovedVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedVGG16, self).__init__()

        # Add Batch Normalization after each convolution layer to improve feature normalization
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4 - Add dilated convolutions for larger receptive field
            nn.Conv2d(256, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5 - Add Squeeze-and-Excitation block for channel attention
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Squeeze-and-Excitation block
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 512 // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // 16, 512, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size after feature extraction
        self._calculate_flatten_size()

        # Improved classifier with additional regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 4096),
            nn.BatchNorm1d(4096),  # Add BatchNorm to FC layers
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Increase dropout for better regularization
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def _calculate_flatten_size(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            x = self.features(x)
            # Apply SE attention
            se_weight = self.se_block(x)
            x = x * se_weight
            # Apply spatial attention
            spatial_weight = self.spatial_attention(x)
            x = x * spatial_weight
            # Final pooling
            x = self.pool(x)
            self.flatten_size = x.numel() // x.shape[0]

    def forward(self, x):
        # Feature extraction
        x = self.features(x)

        # Channel attention with SE block
        se_weight = self.se_block(x)
        x = x * se_weight

        # Spatial attention
        spatial_weight = self.spatial_attention(x)
        x = x * spatial_weight

        # Final pooling and classification
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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

    model = ImprovedVGG16(num_classes=num_classes).to(device)
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

