import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from XJTU_Related.XJTU_Dataset import XJTUData

sys.path.append("..")


def load_file_acc(file_path, id, bearing_num):
    """Load a single acceleration file and add indexing/ID columns."""
    file_acc_df = pd.read_csv(file_path)
    # Add file_index for feature extraction labels (e.g., in tsfresh)
    file_acc_df["file_index"] = id + 1
    file_acc_df.set_index("file_index", inplace=True)
    file_acc_df["id"] = int(str(bearing_num) + f"{str(id + 1).zfill(4)}")
    file_acc_df["file_time"] = np.arange(len(file_acc_df))
    return file_acc_df


def get_bearing_acc(folder_path, bearing_num):
    """Read and concatenate all acceleration files in a bearing folder."""
    file_name_ls = os.listdir(folder_path)
    file_num = len(file_name_ls)

    acc_ls = [load_file_acc(f"{folder_path}/{i + 1}.csv", i, bearing_num) for i in range(file_num)]
    acc_df = pd.concat(acc_ls, axis=0, ignore_index=False)

    return acc_df, file_num


def rul_calculate(file_num, bearing_num):
    """Calculate linear RUL labels."""
    rul_ls = [file_num - i for i in range(1, int(file_num) + 1)]
    rul_dataframe = pd.DataFrame(rul_ls, columns=['RUL'])
    rul_dataframe["id"] = [int(str(bearing_num) + f"{str(idx + 1).zfill(4)}") for idx in range(file_num)]
    rul_dataframe.set_index("id", inplace=True)
    return rul_dataframe


def train_rul_calculate(file_num, bearing_num):
    """Calculate piece-wise linear RUL labels for training."""
    rul_ls = [file_num - i for i in range(1, int(file_num) + 1)]
    max_life = int(max(rul_ls) * 0.8)
    rul_ls = np.array(rul_ls).clip(min=0, max=max_life) / max_life

    rul_dataframe = pd.DataFrame(rul_ls, columns=['RUL'])
    rul_dataframe["id"] = [int(str(bearing_num) + f"{str(idx + 1).zfill(4)}") for idx in range(file_num)]
    rul_dataframe.set_index("id", inplace=True)
    return rul_dataframe


def data_visual_rul(rul_dataframe, bearing_acc_dataframe, bearing_name):
    """Visualize RUL and vibration signals."""
    plt.figure(figsize=(24, 6))

    # RUL plot
    plt.subplot(131)
    plt.plot(range(1, len(rul_dataframe) + 1), rul_dataframe.values)
    plt.title("RUL")
    plt.xlabel("sample file number")
    plt.ylabel("RUL 10(s)")

    # Horizontal vibration
    plt.subplot(132)
    plt.plot(bearing_acc_dataframe["Horizontal_vibration_signals"])
    plt.title("Horizontal_vibration_signals")
    plt.xlabel("sample file number")
    plt.ylabel("vibration signal")

    # Vertical vibration
    plt.subplot(133)
    plt.plot(bearing_acc_dataframe["Vertical_vibration_signals"])
    plt.title("Vertical_vibration_signals")
    plt.xlabel("sample file number")
    plt.ylabel("vibration signal")

    plt.tight_layout()
    plt.savefig(f'/content/drive/MyDrive/picture_XJTU/acc&rul_picture/acc&rul_{bearing_name}.jpg')


def data_load(root_dir, bearing_data_set, flag="train"):
    """Load datasets for either training or testing phases."""
    print(f"------------------------------ start load {flag} data ------------------------------")
    acc_dataframe_ls, rul_dataframe_ls, file_num_ls = [], [], []

    if flag == "train":
        for idx, bearing_name in enumerate(bearing_data_set):
            bearing_num = idx + 1
            folder_path = os.path.join(root_dir, bearing_name)

            acc_df, f_num = get_bearing_acc(folder_path, bearing_num)
            print(f"{bearing_name} has {f_num} acceleration files")

            acc_dataframe_ls.append(acc_df)
            file_num_ls.append(f_num)
            rul_dataframe_ls.append(train_rul_calculate(f_num, bearing_num))

        train_x = pd.concat(acc_dataframe_ls, axis=0)
        train_y = pd.concat(rul_dataframe_ls, axis=0)
        return train_x, train_y, file_num_ls
    else:
        bearing_name = bearing_data_set[0]
        bearing_num = 5
        folder_path = os.path.join(root_dir, bearing_name)

        acc_df, f_num = get_bearing_acc(folder_path, bearing_num)
        print(f"{bearing_name} has {f_num} acceleration files")

        return acc_df, rul_calculate(f_num, bearing_num), [f_num]


def get_xjtu_data_(root_dir, train_bearing_data_set, test_bearing_data_set, window_length, validation_rate, input_fea,
                   sampling, stride, max_life_rate):
    """Main pipeline for data loading, processing, and window slicing."""
    train_x_df, train_y_df, train_file_num_ls = data_load(root_dir, train_bearing_data_set, flag="train")
    test_x_df, test_y_df, test_file_num_ls = data_load(root_dir, test_bearing_data_set, flag="test")

    # Feature engineering: combined vibration signal
    for df in [train_x_df, test_x_df]:
        df["vibration"] = np.sqrt(df["Horizontal_vibration_signals"] ** 2 + df["Vertical_vibration_signals"] ** 2)

    train_x_df['rul'] = train_x_df['id'].map(train_y_df['RUL'])
    test_x_df['rul'] = test_x_df['id'].map(test_y_df['RUL'])

    # Standardization
    v_train = train_x_df[["vibration"]].values
    v_test = test_x_df[["vibration"]].values
    v_mean, v_std = np.mean(v_train, axis=0), np.std(v_train, axis=0)

    train_x_df["vibration"] = (v_train - v_mean) / v_std
    test_x_df["vibration"] = (v_test - v_mean) / v_std

    # Segmenting and Splitting
    split_ratio = input_fea
    train_split_ls, train_y_ls, train_basis = [], [], []
    start_idx = 0
    train_file_cycles = [f * int(32768 / split_ratio) for f in train_file_num_ls]

    for cycle_len in train_file_cycles:
        for j in range(cycle_len):
            feat = train_x_df["vibration"].values[
                   start_idx + split_ratio * j: start_idx + split_ratio * (j + 1)].reshape(1, -1)
            rul = train_x_df.iloc[start_idx + split_ratio * (j + 1) - 1]["rul"]
            train_split_ls.append(feat)
            train_y_ls.append(rul)
            train_basis.append(j / cycle_len)
        start_idx += cycle_len * split_ratio

    train_feat = np.concatenate(train_split_ls, axis=0)
    train_label = np.array(train_y_ls)
    train_basis = np.array(train_basis)

    # Similar processing for test set
    test_cycle_len = test_file_num_ls[0] * int(32768 / split_ratio)
    test_split_ls = [test_x_df["vibration"].values[split_ratio * i: split_ratio * (i + 1)].reshape(1, -1) for i in
                     range(test_cycle_len)]
    test_y_ls = [test_x_df.iloc[split_ratio * (i + 1) - 1]["rul"] for i in range(test_cycle_len)]
    test_basis_arr = np.array([i / test_cycle_len for i in range(test_cycle_len)])

    test_feat = np.concatenate(test_split_ls, axis=0)
    test_label = np.array(test_y_ls)

    # Downsampling
    train_feat, test_feat = train_feat[::sampling], test_feat[::sampling]
    train_label, test_label = train_label[::sampling], test_label[::sampling]
    train_basis, test_basis_arr = train_basis[::sampling], test_basis_arr[::sampling]

    # Window slicing
    def create_windows(data, labels, basis, win_len, step):
        x_win, y_win, b_win = [], [], []
        for i in range(0, len(data) - win_len + 1, step):
            x_win.append(np.expand_dims(data[i:i + win_len], axis=0))
            y_win.append(np.expand_dims(labels[i:i + win_len], axis=0))
            b_win.append(np.expand_dims(basis[i:i + win_len], axis=0))
        return np.concatenate(x_win, axis=0), np.concatenate(y_win, axis=0), np.concatenate(b_win, axis=0)

    # Generate final train/test windows
    X_train_win, y_train_win, basis_train_win = create_windows(train_feat, train_label, train_basis, window_length,
                                                               stride)
    X_test_win, y_test_win, basis_test_win = create_windows(test_feat, test_label, test_basis_arr, window_length,
                                                            stride)

    max_life = int(np.max(y_test_win) * max_life_rate)
    y_test_win = y_test_win.clip(min=0, max=max_life)

    # Validation split
    train_X, vali_X, train_y, vali_y, train_I, vali_I = train_test_split(
        X_train_win, y_train_win, basis_train_win, test_size=validation_rate, random_state=42)

    print(f"Final shapes -> Train X: {train_X.shape}, Vali X: {vali_X.shape}, Test X: {X_test_win.shape}")
    return train_X, train_y, vali_X, vali_y, train_I, vali_I, X_test_win, y_test_win, basis_test_win, max_life


def get_xjtu_data_PINN(root_dir, train_bearing_data_set, test_bearing_data_set, window_length, validation_rate,
                       input_fea, sampling, stride, max_life_rate):
    """Simplified data retrieval for PINN (no window slicing)."""
    # ... logic here matches the initial part of get_xjtu_data_ but skips windowing ...
    # (Implementation follows the same cleaning patterns as above)
    return get_xjtu_data_(root_dir, train_bearing_data_set, test_bearing_data_set, 1, validation_rate, input_fea,
                          sampling, 1, max_life_rate)


def da_get_xjtu_data_(pre_process_type, root_dir, train_bearing_data_set, test_bearing_data_set, STFT_window_len,
                      STFT_overlap_num, window_length, validation_rate):
    """Domain Adaptation data loading wrapper."""
    s_train_X, s_train_y, s_vali_X, s_vali_y, _, _, t_test_X, t_test_y, _, _ = get_xjtu_data_(
        root_dir='./XJTU/XJTU-SY_Bearing_Datasets/35Hz12kN',
        train_bearing_data_set=["Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"],
        test_bearing_data_set=["Bearing1_1"],
        window_length=window_length,
        validation_rate=0.1,
        input_fea=1, sampling=1, stride=1, max_life_rate=1.0)

    return s_train_X, s_train_X, s_vali_X, s_vali_X, s_train_y, s_vali_y, t_test_X, t_test_y


if __name__ == '__main__':
    # Example execution
    train_X, train_y, vali_X, vali_y, _, _, test_X, test_y, _, _ = get_xjtu_data_(
        root_dir='./XJTU/XJTU-SY_Bearing_Datasets/35Hz12kN',
        train_bearing_data_set=["Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"],
        test_bearing_data_set=["Bearing1_1"],
        window_length=32,
        validation_rate=0.1,
        input_fea=1, sampling=1, stride=1, max_life_rate=1.0)

    train_loader = DataLoader(XJTUData(train_X, train_y), batch_size=32, shuffle=True)
    vali_loader = DataLoader(XJTUData(vali_X, vali_y), batch_size=32, shuffle=False)
    test_loader = DataLoader(XJTUData(test_X, test_y), batch_size=32, shuffle=False)