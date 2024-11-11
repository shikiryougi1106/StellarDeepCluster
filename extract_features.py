import torch
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def extract_features(model, dataloader):
    model.eval()
    features, labels_list = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model.base_model.features(images)
            outputs = nn.functional.adaptive_avg_pool2d(outputs, (1, 1)).view(outputs.size(0), -1)
            features.append(outputs.cpu())
            labels_list.extend(labels.cpu().numpy())
    return torch.cat(features).numpy(), np.array(labels_list)

# 后面使用 features 和 labels 进行 K-Means 聚类
