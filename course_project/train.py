import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# --- 1. Определение модели (должна быть идентична model.py) ---
class GenreCNN(nn.Module):
    """Простая CNN для классификации жанров по спектрограммам 128x128"""
    def __init__(self, num_classes=10):
        super(GenreCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- 2. Датасет и трансформации ---
# Преобразования изображений (должны совпадать с model.py)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- 3. Загрузка данных ---
# Укажите путь к папке с датасетом (например, 'путь/к/вашей/папке/Data/spectrograms/')
data_dir = '/Users/elzaalpnva/Desktop/мага (МАИ)/УХов/label-studio-ml-backend/my_music_classifier/Data/images_original'  # <-- ИЗМЕНИТЕ ЭТУ ПЕРЕМЕННУЮ
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

all_files = []
all_labels = []
for idx, genre in enumerate(genres):
    genre_path = os.path.join(data_dir, genre)
    if not os.path.exists(genre_path):
        print(f'Предупреждение: Папка {genre_path} не найдена. Пропускаем жанр.')
        continue
    for fname in os.listdir(genre_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_files.append(os.path.join(genre_path, fname))
            all_labels.append(idx)

# Разделение на обучающую и валидационную выборки
train_files, val_files, train_labels, val_labels = train_test_split(
    all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# Создание DataLoader'ов
train_dataset = SpectrogramDataset(train_files, train_labels, transform=transform)
val_dataset = SpectrogramDataset(val_files, val_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- 4. Обучение модели ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Используемое устройство: {device}')
model = GenreCNN(num_classes=len(genres)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
best_val_loss = float('inf')

print('Начинаем обучение...')
for epoch in range(num_epochs):
    # --- Обучение ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)
    train_loss = train_loss / len(train_dataset)
    train_acc = train_correct.double() / len(train_dataset)

    # --- Валидация ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)
    val_loss = val_loss / len(val_dataset)
    val_acc = val_correct.double() / len(val_dataset)

    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
          f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    # Сохраняем лучшую модель
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model_weights.pth')
        print(f'Лучшая модель сохранена (вариант {epoch+1})')

print('Обучение завершено!')