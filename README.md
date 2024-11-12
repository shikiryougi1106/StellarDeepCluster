# StellarDeepCluster
Semi-Supervised Model for Spectral Classification 

## 项目简介

本项目旨在开发一个高效的半监督光谱分类系统，利用深度学习技术结合有限的有标签数据和大量的无标签数据，提升图像分类的准确率和模型的泛化能力。系统采用预训练的卷积神经网络（ `MobileNetV2`）作为基础模型，并通过伪标签生成和一致性正则化方法实现半监督学习。

## 技术栈

- **编程语言**: Python 3.8
- **深度学习框架**: PyTorch
- **主要库**:
  - `numpy`, `pandas` - 数据处理
  - `PIL` - 图像处理
  - `scikit-learn` - 数据预处理与评估
  - `torchvision` - 图像变换与预训练模型
  - `scipy` - 优化算法

## 数据集
使用图像数据集为`SDSS release`

## 总结
进行了`FixMatch` 半监督学习算法 `deepcluster`聚类算法，用于图像分类任务。<br>
在实际应用中，标记数据往往稀缺且昂贵，而未标记数据大量存在。`FixMatch` 通过结合一致性正则化和伪标签生成，有效利用未标记数据，提升模型性能.

# 1. 数据准备和预处理：

数据读取：从 `CSV` 文件中读取图像路径和对应的类别标签。<br>
标签编码：使用 `LabelEncoder` 将文本标签编码为整数，方便模型训练。<br>
数据划分：按照类别比例，使用 `train_test_split` 将数据集划分为有标签数据（如 10%）和无标签数据（如 90%），确保类别分布一致。<br>
未标记数据处理：将未标记数据的标签设为 'unlabeled'，编码标签设为 -1，表示无效或未知类别。<br>
# 2. 数据增强策略：

自定义转换：定义 `GrayTo3Channels` 类，将灰度图像转换为 3 通道的 RGB 图像，以适应预训练模型的输入要求。<br>
弱增强（`Weak Augmentation`）：<br>
用于生成伪标签，包含基本的数据增强操作，如调整尺寸、随机水平翻转等。<br>
保持图像的主要特征，确保模型对弱增强图像的预测具有较高的可信度。<br>
强增强（`Strong Augmentation`）：<br>
用于一致性训练，包含更复杂的数据增强操作，如随机旋转、随机裁剪、颜色抖动等。<br>
增加数据的多样性，提升模型的鲁棒性。<br>
# 3. 数据集和数据加载器：

有标签数据集（`LabeledImageDataset`）：<br>
继承自 `torch.utils.data.Dataset`，用于加载有标签的数据。<br>
应用弱增强，返回图像和对应的标签。<br>
无标签数据集（`UnlabeledImageDataset`）：<br>
用于加载无标签的数据，同时返回弱增强和强增强的图像。<br>
弱增强图像用于生成伪标签，强增强图像用于一致性训练。<br>
数据加载器（`DataLoader`）：<br>
创建有标签和无标签数据的 `DataLoader`，支持批量数据加载和随机打乱。<br>
# 4. 模型构建：

预训练模型：使用 `torchvision.models` 中的预训练模型 `MobileNetV2`。<br>
模型修改：
替换最后的全连接层，以适应数据集的类别数量。<br>
添加 `Dropout` 层，防止过拟合，提高模型的泛化能力。<br>
模型实例化：根据有标签数据的类别数量，初始化模型并移动到指定设备（CPU 或 GPU）。<br>
# 5. 训练过程：

损失函数和优化器：<br>
使用交叉熵损失函数 `nn.CrossEntropyLoss()`。<br>
使用 `Adam` 优化器，学习率设为 0.001。<br>
`FixMatch` 训练参数：<br>
置信度阈值（`confidence_threshold`）：用于筛选高置信度的伪标签样本（如 0.95）。<br>
无监督损失权重（`lambda_u`）：控制有监督和无监督损失的平衡（如 0.1）。<br>
训练轮数（`num_epochs`）：设置训练的总轮数（如 15,100）。<br>
对有标签数据，进行前向传播，计算有监督损失 `loss_supervised`。<br>
记录预测结果，计算训练准确率。<br>
无监督训练（`FixMatch` 核心部分）<br>
从无标签数据加载器中获取弱增强和强增强的图像。<br>
使用模型对弱增强图像进行预测，生成伪标签。<br>
筛选置信度高于阈值的样本，得到高置信度的伪标签和对应的强增强图像。<br>
对强增强图像进行前向传播，计算无监督损失 `loss_unsupervised`。<br>
损失计算和反向传播：<br>
计算总损失 `loss = loss_supervised + lambda_u * loss_unsupervised`。<br>
进行反向传播 `loss.backward()`，更新模型参数 `optimizer.step()`。<br>
日志记录：<br>
记录每个 `epoch` 的平均损失、准确率和耗时，便于监控训练进度。<br>
# 6. 特征提取和聚类评估：

特征提取：
定义 `extract_features` 函数，从模型的特征提取部分获取特征表示。<br>
应用全局平均池化，将特征映射为固定长度的向量。<br>
`K-Means` 聚类：<br>
使用提取的特征进行 `K-Means` 聚类<br>
标签匹配和评估：<br>
使用匈牙利算法（`linear_sum_assignment`）调整聚类标签与真实标签的对应关系，最大化准确率。<br>
计算并打印分类报告，包括精确率、召回率和 F1 分数。<br>




## 半监督学习实现
# 1.数据增强策略
弱数据增强 用于无标签数据，保持图像的基本特征，确保伪标签的可靠性。
 ```python
 weak_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    GrayTo3Channels(),
    transforms.ToTensor(),
])
```
强数据增强 用于有标签数据，增加训练样本的多样性，提升模型的鲁棒性。
```python
  strong_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
    transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    GrayTo3Channels(),
    transforms.ToTensor(),
])
```
# 2.数据集定义
# 有标签数据
 ```python
 class LabeledImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.copy().reset_index(drop=True)
        self.transform = transform

        # 标签编码
        self.label_encoder = LabelEncoder()
        self.df['Class_encoded'] = self.label_encoder.fit_transform(self.df['Class'])

        # 过滤不存在的图像文件
        existing_files = [os.path.exists(path) for path in self.df['image_path']]
        self.df = self.df[existing_files].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        image = Image.open(img_path).convert('L')  # 读取为灰度图
        label = self.df.loc[idx, 'Class_encoded']
        if self.transform:
            image = self.transform(image)
        return image, label
```
# 无标签数据
```python
 class UnlabeledImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.copy().reset_index(drop=True)
        self.transform = transform

        # 过滤不存在的图像文件
        existing_files = [os.path.exists(path) for path in self.df['image_path']]
        self.df = self.df[existing_files].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        image = Image.open(img_path).convert('L')  # 读取为灰度图
        if self.transform:
            image = self.transform(image)
        return image
```
# 半监督数据集类
```python
 class SemiSupervisedDataset(Dataset):
    def __init__(self, labeled_df, unlabeled_df, labeled_transform=None, unlabeled_transform=None):
        self.labeled_dataset = LabeledImageDataset(labeled_df, transform=labeled_transform)
        self.unlabeled_dataset = UnlabeledImageDataset(unlabeled_df, transform=unlabeled_transform)

    def __len__(self):
        return max(len(self.labeled_dataset), len(self.unlabeled_dataset))

    def __getitem__(self, idx):
        labeled_data = self.labeled_dataset[idx % len(self.labeled_dataset)]
        unlabeled_data = self.unlabeled_dataset[idx % len(self.unlabeled_dataset)]
        return labeled_data, unlabeled_data
```
# 模型定义
```python
# 定义深度卷积神经网络，使用预训练的 MobileNetV2作为基础
class DeepCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepCNN, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)

# 初始化模型、损失函数和优化器
num_classes = len(df_labeled['Class'].unique())
model = DeepCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# FixMatch 训练参数
confidence_threshold = 0.95  # 置信度阈值
lambda_u = 0.1       # 无监督损失的权重
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    start_time = time.time()

    unlabeled_iter = iter(unlabeled_dataloader)

    for images_labeled, labels in labeled_dataloader:
        images_labeled = images_labeled.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 处理有标签的数据
        outputs_labeled = model(images_labeled)
        loss_supervised = criterion(outputs_labeled, labels)

        # 记录预测结果
        _, preds = torch.max(outputs_labeled, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        # 获取一批未标记的数据
        try:
            weak_images, strong_images = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_dataloader)
            weak_images, strong_images = next(unlabeled_iter)

        weak_images = weak_images.to(device)
        strong_images = strong_images.to(device)

        # 对弱增强的未标记数据进行预测，生成伪标签
        with torch.no_grad():
            outputs_weak = model(weak_images)
            probabilities = torch.softmax(outputs_weak, dim=1)
            max_probs, pseudo_labels = torch.max(probabilities, dim=1)

        # 筛选高置信度的样本
        mask = max_probs >= confidence_threshold
        selected_strong_images = strong_images[mask]
        selected_pseudo_labels = pseudo_labels[mask]

        if len(selected_strong_images) > 0:
            # 对强增强的未标记数据进行训练
            outputs_strong = model(selected_strong_images)
            loss_unsupervised = criterion(outputs_strong, selected_pseudo_labels)
            loss = loss_supervised + lambda_u * loss_unsupervised
        else:
            loss = loss_supervised

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
```
# 性能评估
```python
 def evaluate(model, device, test_loader, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
```
## 训练结果
```bash
Epoch [1/25], Loss: 0.5185, Accuracy: 0.8210, Pseudo-label Ratio: 0.5103, Time: 61.26 seconds
Epoch [2/25], Loss: 0.2798, Accuracy: 0.8987, Pseudo-label Ratio: 0.6371, Time: 70.36 seconds
Epoch [3/25], Loss: 0.2168, Accuracy: 0.9228, Pseudo-label Ratio: 0.7027, Time: 70.41 seconds
Epoch [4/25], Loss: 0.2119, Accuracy: 0.9205, Pseudo-label Ratio: 0.7156, Time: 65.99 seconds
Epoch [5/25], Loss: 0.1624, Accuracy: 0.9411, Pseudo-label Ratio: 0.7790, Time: 56.80 seconds
Epoch [6/25], Loss: 0.1524, Accuracy: 0.9527, Pseudo-label Ratio: 0.7830, Time: 55.66 seconds
Epoch [7/25], Loss: 0.1162, Accuracy: 0.9594, Pseudo-label Ratio: 0.8063, Time: 56.49 seconds
Epoch [8/25], Loss: 0.1065, Accuracy: 0.9661, Pseudo-label Ratio: 0.8406, Time: 59.77 seconds
Epoch [9/25], Loss: 0.1097, Accuracy: 0.9661, Pseudo-label Ratio: 0.8455, Time: 60.22 seconds
Epoch [10/25], Loss: 0.1310, Accuracy: 0.9580, Pseudo-label Ratio: 0.7929, Time: 55.76 seconds
Epoch [11/25], Loss: 0.1296, Accuracy: 0.9531, Pseudo-label Ratio: 0.8379, Time: 56.50 seconds
Epoch [12/25], Loss: 0.1322, Accuracy: 0.9567, Pseudo-label Ratio: 0.8330, Time: 56.81 seconds
Epoch [13/25], Loss: 0.0948, Accuracy: 0.9750, Pseudo-label Ratio: 0.8228, Time: 60.36 seconds
Epoch [14/25], Loss: 0.0772, Accuracy: 0.9804, Pseudo-label Ratio: 0.8527, Time: 57.05 seconds
Epoch [15/25], Loss: 0.0842, Accuracy: 0.9759, Pseudo-label Ratio: 0.8866, Time: 56.65 seconds
Epoch [16/25], Loss: 0.0658, Accuracy: 0.9835, Pseudo-label Ratio: 0.8629, Time: 56.45 seconds
Epoch [17/25], Loss: 0.0652, Accuracy: 0.9826, Pseudo-label Ratio: 0.8768, Time: 57.66 seconds
Epoch [18/25], Loss: 0.0644, Accuracy: 0.9862, Pseudo-label Ratio: 0.8866, Time: 56.47 seconds
Epoch [19/25], Loss: 0.0686, Accuracy: 0.9808, Pseudo-label Ratio: 0.8634, Time: 56.53 seconds
Epoch [20/25], Loss: 0.0712, Accuracy: 0.9821, Pseudo-label Ratio: 0.8857, Time: 56.76 seconds
Epoch [21/25], Loss: 0.0474, Accuracy: 0.9888, Pseudo-label Ratio: 0.8857, Time: 56.47 seconds
Epoch [22/25], Loss: 0.0468, Accuracy: 0.9906, Pseudo-label Ratio: 0.8991, Time: 56.66 seconds
Epoch [23/25], Loss: 0.0788, Accuracy: 0.9799, Pseudo-label Ratio: 0.8955, Time: 56.66 seconds
Epoch [24/25], Loss: 0.0635, Accuracy: 0.9821, Pseudo-label Ratio: 0.8661, Time: 56.86 seconds
Epoch [25/25], Loss: 0.0457, Accuracy: 0.9906, Pseudo-label Ratio: 0.8884, Time: 55.89 seconds
训练完成！
              precision    recall  f1-score   support

      GALAXY       1.00      0.99      1.00       727
         QSO       0.99      1.00      1.00      1146
        STAR       1.00      0.98      0.99       367

    accuracy                           1.00      2240
   macro avg       1.00      0.99      0.99      2240
weighted avg       1.00      1.00      1.00      2240
```


## 附录
# 附录A：使用的半监督学习算法简介
`Pseudo-Labeling`：通过模型预测生成伪标签，结合有标签数据共同训练。<br>
`FixMatch`：结合一致性正则化和伪标签生成，使用高置信度的伪标签进行训练。<br>
`MixMatch`：通过数据增强、标签平滑和混合数据增强策略，提高半监督学习效果。<br>
`deepcluster`: 使用`deepcluster`聚类算法,进行聚类。<br>
# 附录B: 使用的半监督学习模型算法
伪标签生成 (`Pseudo-Labeling`): 使用无标签数据的模型预测值作为伪标签，设置一个置信度阈值，仅选择高置信度样本加入训练，以保证伪标签的可靠性。<br>
一致性正则化 (`Consistency Regularization`): 在无标签数据上应用弱数据增强后进行伪标签生成，接着应用强数据增强并再次预测。对于这些增强后的样本，通过损失函数让模型保持一致性，增强模型对数据扰动的鲁棒性。<br>
# 附录C: `deepcluster` 介绍
`DeepCluster`结合了两部分：无监督聚类和深度神经网络。它提出了一种端到端的方法，用于共同学习深度神经网络的参数及其表示的集群分配。特征是迭代生成和聚类的，以获得训练模型和标签作为输出工件。

![deepcluster](https://amitness.com/posts/images/deepcluster-pipeline.gif)

如上图所示，拍摄了无标签的图像，并应用了增强功能。然后，`AlexNet`或`VGG-16`等`ConvNet`架构被用作特征提取器。最初，`ConvNet`是用随机权重初始化的，我们在最终分类头之前从图层中获取特征向量。然后，PCA用于减少特征向量的尺寸，归一化。最后，处理的特征被传递到K-means，以获得每个图像的集群分配。

这些集群分配用作伪标签，ConvNet经过训练来预测这些集群。交叉熵损失用于衡量模型的性能。该模型训练了100个`epoch`，聚类步骤每个`epoch`发生一次。最后，我们可以将学到的表示用于下游任务。

