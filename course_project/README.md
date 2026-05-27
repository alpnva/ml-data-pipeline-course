# Инструкция по использованию системы классификации музыкальных жанров

## 1. Общая информация

Система предназначена для классификации музыкальных жанров по спектрограммам аудио.

Поддерживаемые классы:

* blues
* classical
* country
* disco
* hiphop
* jazz
* metal
* pop
* reggae
* rock

---

## 2. Экспорт размеченных данных из Label Studio

1. Открыть проект
2. Перейти в **Data Manager**
3. Нажать **Export**
4. Выбрать формат **JSON**

Пример структуры файла:

```json
[
  {
    "data": {
      "image": "images/blues_001.png"
    },
    "annotations": [
      {
        "result": [
          {
            "value": {
              "choices": ["blues"]
            }
          }
        ]
      }
    ]
  }
]
```

---

## 3. Конвертация JSON в датасет

### 3.1 Скрипт подготовки данных

```python
import json
import os
import shutil

INPUT_JSON = "export.json"
OUTPUT_DIR = "dataset"

with open(INPUT_JSON) as f:
    data = json.load(f)

for item in data:
    image_path = item['data']['image']
    label = item['annotations'][0]['result'][0]['value']['choices'][0]

    class_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(class_dir, exist_ok=True)

    shutil.copy(image_path, class_dir)
```

После выполнения структура:

```
dataset/
  blues/
  rock/
  jazz/
  ...
```

---

## 4. Обучение модели

### 4.1 Определение модели

```python
import torch.nn as nn

class GenreCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

---

### 4.2 Трансформации (обязательное совпадение)

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

### 4.3 Пример обучения

```python
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

dataset = ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = GenreCNN(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "model_weights.pth")
```

---

## 5. Инференс (использование модели)

### 5.1 Загрузка модели

```python
import torch
from PIL import Image

model = GenreCNN(num_classes=10)
model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
model.eval()
```

---

### 5.2 Предсказание для одного изображения

```python
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred
```

---

### 5.3 Получение названия жанра

```python
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

genre = genres[predict_image("test.png")]
print(genre)
```

---

## 6. Интеграция с Label Studio

ML backend реализует метод:

```python
def predict(self, tasks):
    # загрузка изображения
    # трансформация
    # инференс
    # возврат жанра
```

Формат ответа:

```json
{
  "result": [{
    "from_name": "genre",
    "to_name": "image",
    "type": "choices",
    "value": {
      "choices": ["rock"]
    }
  }]
}
```

---

## 7. Обновление модели

1. Разметить новые данные в Label Studio
2. Экспортировать JSON
3. Обновить датасет
4. Переобучить модель
5. Заменить `model_weights.pth`
6. Перезапустить ML backend

---

## 8. Запуск ML backend

```bash
label-studio-ml start .
```

---

## 9. Зависимости

```bash
pip install torch torchvision pillow label-studio-ml
```

---

## 10. Важные замечания

* Трансформации должны совпадать при обучении и инференсе
* Архитектура модели не должна изменяться
* Классы должны совпадать с Label Studio
* Пути к изображениям должны быть корректными
