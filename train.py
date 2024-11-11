import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from data_processing import LabeledImageDataset, UnlabeledImageDataset, weak_transform, strong_transform
from model import DeepCNN
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, labeled_dataloader, unlabeled_dataloader, criterion, optimizer, num_epochs, confidence_threshold, lambda_u):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_labels, all_preds = [], []
        start_time = time.time()

        unlabeled_iter = iter(unlabeled_dataloader)

        for images_labeled, labels in labeled_dataloader:
            images_labeled, labels = images_labeled.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs_labeled = model(images_labeled)
            loss_supervised = criterion(outputs_labeled, labels)
            _, preds = torch.max(outputs_labeled, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            try:
                weak_images, strong_images = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_dataloader)
                weak_images, strong_images = next(unlabeled_iter)

            weak_images, strong_images = weak_images.to(device), strong_images.to(device)

            with torch.no_grad():
                outputs_weak = model(weak_images)
                probabilities = torch.softmax(outputs_weak, dim=1)
                max_probs, pseudo_labels = torch.max(probabilities, dim=1)

            mask = max_probs >= confidence_threshold
            selected_strong_images = strong_images[mask]
            selected_pseudo_labels = pseudo_labels[mask]

            if len(selected_strong_images) > 0:
                outputs_strong = model(selected_strong_images)
                loss_unsupervised = criterion(outputs_strong, selected_pseudo_labels)
                loss = loss_supervised + lambda_u * loss_unsupervised
            else:
                loss = loss_supervised

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(labeled_dataloader):.4f}, '
              f'Accuracy: {accuracy:.4f}, Time: {time.time() - start_time:.2f}s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    # 示例加载数据和设置
    df_labeled = ...  # 读取有标签的数据
    df_unlabeled = ...  # 读取无标签数据
    labeled_dataset = LabeledImageDataset(df=df_labeled, transform=weak_transform)
    unlabeled_dataset = UnlabeledImageDataset(df=df_unlabeled, weak_transform=weak_transform, strong_transform=strong_transform)

    labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=True)

    model = DeepCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_model(model, labeled_dataloader, unlabeled_dataloader, criterion, optimizer, args.epochs, confidence_threshold=0.95, lambda_u=0.13)
