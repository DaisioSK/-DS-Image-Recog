# define the CNN architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 1. conv layer 1: (1, 28, 28) → (32, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # pooling layer

        # 2. conv layer 2: (32, 14, 14) →  (64, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 3. conv layer 3: (64, 7, 7) →  (128, 7, 7)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 4. full conn layer: (128*7*7) → 128
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)  # drop out to prevent overfitting
        self.fc2 = nn.Linear(128, 62) 

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 + ReLU + Pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 + ReLU + Pooling
        x = F.relu(self.bn3(self.conv3(x)))  # Conv3 + ReLU

        x = x.view(-1, 128 * 7 * 7) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.fc2(x)  
        return x


def train_CNN(model, model_path, train_loader, val_loader, num_epochs=10, patience=3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf') 
    counter = 0  
    current_epoch = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    if os.path.exists(model_path):
        print(f"✅ found {model_path}, loading model ...")
        pretrained_model = torch.load(model_path, map_location=device)
        model.load_state_dict(pretrained_model['model_state_dict'])
        optimizer.load_state_dict(pretrained_model['optimizer_state_dict'])

        train_losses = pretrained_model.get('train_losses', [])
        train_accs = pretrained_model.get('train_accs', [])
        val_losses = pretrained_model.get('val_losses', [])
        val_accs = pretrained_model.get('val_accs', [])
        current_epoch = pretrained_model.get('epoch', 0)
        best_val_loss = pretrained_model.get('best_val_loss', float('inf'))
        counter = pretrained_model.get('counter', 0)

        if current_epoch >= num_epochs:
            print(f"✅ finished training {model_path} with {current_epoch} epoches. No action needed.")
            return train_losses, train_accs, val_losses, val_accs
        else:
            print(f"🔄 continue traiing {num_epochs - current_epoch} epoches...")

    else:
        print(f"⚠️ {model_path} not found, {num_epochs} epoches left...")

    for epoch in range(current_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0 
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'counter': counter
            }, model_path)
            print(f"model updated to {model_path}...")
        else:
            counter += 1
            print(f"early stopping counter: {counter}/{patience}")

        if counter >= patience:
            print("early stopped!")
            break

    print("training completed 🚀")
    return train_losses, train_accs, val_losses, val_accs
