import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from tensorboard.compat.tensorflow_stub.dtypes import int64
import seaborn as sns
from New_CMAPSS_Dataset import New_CMAPSSData
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def creat_label(left_data,right_data):
    zero_array = np.zeros((len(left_data), 1))
    ones_array = np.zeros((len(right_data),1))

def kl_divergence(p, q):
    """计算KL散度"""
    p = p / np.sum(p)
    q = q / np.sum(q)
    return entropy(p, q)


def split_changed_point(train_values , sequence_length):

    if isinstance(train_values, pd.DataFrame):
        train_values = train_values.values
    min_num, max_num = np.min(train_values[:, 0]), np.max(train_values[:, 0])
    sample = []
    for i in range(int(min_num.tolist()), int((max_num+1).tolist())):
        sample.append(train_values[train_values[:, 0]==i])
    window_size = 60

    safe_sample = []
    change_points = []
    scatters = []
    for n,data in enumerate(sample):
        kl_changes = []

        for i in range(len(data) - 2 * window_size + 1):
            window1 = data[i:i + window_size]
            window2 = data[i + window_size:i + 2 * window_size]

            hist1, bin_edges1 = np.histogram(window1, bins=10, density=True)
            hist2, bin_edges2 = np.histogram(window2, bins=10, density=True)

            kl = kl_divergence(hist1, hist2)
            kl_changes.append(kl)

        kl_changes = np.array(kl_changes)
        change_point = np.argmax(kl_changes) + window_size
        scatter = int(change_point) / len(data)
        scatters.append(scatter)
        change_points.append(change_point)

        left_datas = []
        right_datas = []
        left_data = data[:change_point]
        right_data = data[change_point:]
        for m in range(len(left_data)-2*sequence_length+1):
            left_datas.append(left_data[m:m+sequence_length])
        for j in range(len(right_data)-2*sequence_length+1):
            right_datas.append(data[j:j+sequence_length])
        left_datas = np.array(left_datas)
        right_datas = np.array(right_datas)

def MinMax_normalize(data_3d):

    sample,time,feature = data_3d.shape
    data_2d = data_3d.reshape(-1, feature)
    data_2d = pd.DataFrame(data_2d)
    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']
    data_2d.columns = column_name

    col_to_drop = ['setting2','setting3','s1','s5','s6','s10','s16','s18','s19']
    data_2d = data_2d.drop(col_to_drop, axis=1)

    data_2d['setting1'] = data_2d['setting1'].round(1)
    grouped_train = data_2d.groupby('setting1')
    scaler = MinMaxScaler()
    train_normalized = pd.DataFrame(columns=data_2d.columns[3:])
    for train_idx, train in grouped_train:

        scaled_train = scaler.fit_transform(train.iloc[:, 3:])
        scaled_train_combine = pd.DataFrame(
            data=scaled_train,
            index=train.index,
            columns=data_2d.columns[3:])
        train_normalized = pd.concat([train_normalized, scaled_train_combine])
    train_normalized = train_normalized.sort_index()
    data_2d.iloc[:, 3:-1] = train_normalized.iloc[:, :]
    data_2d = data_2d.drop('setting1', axis=1)
    data_2d = data_2d.drop('engine_id' , axis=1)
    data_2d = data_2d.drop('cycle', axis=1)
    data_2d = np.array(data_2d)
    _, feature1 = data_2d.shape
    normalized_3d = data_2d.reshape(sample,time,feature1)
    return normalized_3d

def merge_npz_files(input_folder, output_path='merged_data.npz'):

    file_paths = sorted(glob.glob(os.path.join(input_folder, '*.npz')))

    if not file_paths:
        raise FileNotFoundError("目标文件夹中未找到npz文件")

    merged_data = []

    # 遍历每个文件并合并数组
    for file_path in file_paths:
        with np.load(file_path) as data:
            merged_data.append(data["arr_0"])

    merged_data = np.concatenate(merged_data, axis=0)#（样本数，时间步，特征数）
    normalize_merged_data = MinMax_normalize(merged_data)

def split_train_val(x_data,y_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=20, shuffle=True)
    return x_train,x_test,y_train,y_test

def merge_data_label(left_data_path,right_data_path,left_label_path,right_label_path):
    left_data = np.load(left_data_path)
    right_data = np.load(right_data_path)
    left_label = np.load(left_label_path)
    right_label = np.load(right_label_path)
    x_data = np.concatenate((left_data["arr_0"],right_data["arr_0"]),axis=0).astype(np.float32)
    y_data = np.concatenate((left_label["arr_0"],right_label["arr_0"]),axis=0).astype(np.int64)
    y_data =y_data.flatten()
    x_train,x_test,y_train,y_test = split_train_val(x_data, y_data)
    train_dataset = New_CMAPSSData(x_train,y_train)
    test_dataset = New_CMAPSSData(x_test,y_test)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        drop_last=False
    )

    model = TimeSeriesMLP(input_dim=20 * 14, num_classes=2).to(device)
    train_cm, test_cm, train_loss, test_loss = train_model(model, train_loader, test_loader, epochs=10)

    plot_confusion_matrix(train_cm, "Train Confusion Matrix")

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("D:/China wyh laboratory/杂事/RUL_classify/train")


def train_model(model, train_loader, test_loader, epochs=50):
    train_losses, test_losses = [], []
    train_matrix = np.zeros((model.num_classes, model.num_classes))
    test_matrix = np.zeros((model.num_classes, model.num_classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    for epoch in range(epochs):

        model.train()
        epoch_train_loss = 0
        all_preds, all_labels = [], []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = epoch_train_loss / len(train_loader)
        train_cm = confusion_matrix(all_labels, all_preds)
        train_acc = np.diag(train_cm).sum() / train_cm.sum()
        train_matrix += train_cm

        model.eval()
        epoch_test_loss = 0
        test_preds, test_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_loss = epoch_test_loss / len(test_loader)
        test_cm = confusion_matrix(test_labels, test_preds)
        test_acc = np.diag(test_cm).sum() / test_cm.sum()
        test_matrix += test_cm

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}\n")

    return train_matrix, test_matrix, train_losses, test_losses




