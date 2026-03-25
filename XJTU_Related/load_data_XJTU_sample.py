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
    """Load acceleration file and append tracking IDs."""
    df = pd.read_csv(file_path)
    # Tag for tsfresh feature extraction
    df["file_index"] = id + 1
    df.set_index("file_index", inplace=True)
    df["id"] = int(str(bearing_num) + f"{str(id + 1).zfill(4)}")
    df["file_time"] = np.arange(len(df))
    return df


def get_bearing_acc(folder_path, bearing_num):
    """Iterate and concatenate all vibration CSVs in a bearing folder."""
    file_list = os.listdir(folder_path)
    file_count = len(file_list)

    acc_list = [load_file_acc(f"{folder_path}/{i + 1}.csv", i, bearing_num) for i in range(file_count)]
    return pd.concat(acc_list, axis=0), file_count


def rul_calculate(file_num, bearing_num):
    """Generate linear RUL labels dataframe."""
    rul_values = [file_num - i for i in range(1, int(file_num) + 1)]
    df = pd.DataFrame(rul_values, columns=['RUL'])
    df["id"] = [int(str(bearing_num) + f"{str(i + 1).zfill(4)}") for i in range(file_num)]
    df.set_index("id", inplace=True)
    return df


def data_visual_rul(rul_df, acc_df, bearing_name):
    """Save visualization plots for RUL and vibration signals."""
    plt.figure(figsize=(24, 6))

    plt.subplot(131)
    plt.plot(range(1, len(rul_df) + 1), rul_df.values)
    plt.title("RUL")
    plt.xlabel("sample file number")
    plt.ylabel("RUL 10(s)")

    plt.subplot(132)
    plt.plot(acc_df["Horizontal_vibration_signals"])
    plt.title("Horizontal Vibration")

    plt.subplot(133)
    plt.plot(acc_df["Vertical_vibration_signals"])
    plt.title("Vertical Vibration")

    plt.tight_layout()
    save_path = f'/content/drive/MyDrive/picture_XJTU/acc&rul_picture/acc&rul_{bearing_name}.jpg'
    plt.savefig(save_path)


def data_load(root_dir, bearing_data_set, flag="train"):
    """Load dataset based on train/test flags."""
    print(f"------------------- Start loading {flag} data -------------------")
    acc_list, rul_list, file_counts = [], [], []

    if flag == "train":
        for i, name in enumerate(bearing_data_set):
            b_num = i + 1
            folder = os.path.join(root_dir, name)
            acc_df, f_num = get_bearing_acc(folder, b_num)

            print(f"{name}: {f_num} files")
            acc_list.append(acc_df)
            file_counts.append(f_num)
            rul_list.append(rul_calculate(f_num, b_num))

        return pd.concat(acc_list), pd.concat(rul_list), file_counts
    else:
        name = bearing_data_set[0]
        folder = os.path.join(root_dir, name)
        acc_df, f_num = get_bearing_acc(folder, 5)  # Default bearing_num 5 for test
        print(f"{name}: {f_num} files")
        return acc_df, rul_calculate(f_num, 5), [f_num]


def get_xjtu_data_(pre_process_type, root_dir, train_bearing_data_set, test_bearing_data_set,
                   STFT_window_len, STFT_overlap_num, window_length, validation_rate):
    """Main processing pipeline: loading, standardization, and window segmentation."""

    # Load raw data (assuming pre-computed vibration files exist)
    train_x_df = pd.read_csv('./XJTU_Related/train_x_vibration', index_col=0)
    test_x_df = pd.read_csv('./XJTU_Related/test_x_vibration', index_col=0)
    _, train_y_df, train_file_num_ls = data_load(root_dir, train_bearing_data_set, flag="train")
    _, test_y_df, test_file_num_ls = data_load(root_dir, test_bearing_data_set, flag="test")

    # Standardization
    v_train, v_test = train_x_df[["vibration"]].values, test_x_df[["vibration"]].values
    v_mean, v_std = np.mean(v_train, axis=0), np.std(v_train, axis=0)
    stand_train = (v_train - v_mean) / v_std
    stand_test = (v_test - v_mean) / v_std

    # Split based on file cycles
    split_ratio = 40
    points_per_file = 32768
    cycles_per_file = int(points_per_file / split_ratio)

    def segment_vibration(data, file_nums):
        segmented = []
        for i in range(len(file_nums)):
            base_idx = i * points_per_file
            for j in range(file_nums[i] * cycles_per_file):
                chunk = data[base_idx + split_ratio * j: base_idx + split_ratio * (j + 1), :].T
                segmented.append(chunk)
        return np.concatenate(segmented, axis=0)

    train_features = segment_vibration(stand_train, train_file_num_ls)

    # Process test features (single bearing logic)
    test_cycles = test_file_num_ls[0] * cycles_per_file
    test_features = np.concatenate(
        [stand_test[split_ratio * i: split_ratio * (i + 1), :].T for i in range(test_cycles)], axis=0)

    # RUL Interpolation logic
    def interpolate_rul(y_df, cycles):
        y_vals = y_df.values.flatten().astype(float)
        # Vectorized interpolation: each original RUL point followed by (cycles-1) interpolated steps
        interp_steps = np.linspace(0, 1, cycles, endpoint=False)
        full_rul = np.array([val + interp_steps for val in y_vals]).flatten()
        return full_rul[:, np.newaxis]

    train_y_interp = interpolate_rul(train_y_df, cycles_per_file)
    test_y_interp = interpolate_rul(test_y_df, cycles_per_file)

    # Sliding window segmentation
    def apply_window(feat, label, win_len):
        x_win, y_win = [], []
        for i in range(len(label) - win_len + 1):
            x_win.append(feat[i: i + win_len, :][np.newaxis, :])
            y_win.append(label[i: i + win_len, :].T)
        return np.concatenate(x_win, axis=0), np.concatenate(y_win, axis=0)

    # Slice train data per bearing to prevent cross-bearing windows
    X_train_list, y_train_list = [], []
    curr_idx = 0
    for f_num in train_file_num_ls:
        total_cycles = f_num * cycles_per_file
        x_s, y_s = apply_window(train_features[curr_idx: curr_idx + total_cycles],
                                train_y_interp[curr_idx: curr_idx + total_cycles], window_length)
        X_train_list.append(x_s)
        y_train_list.append(y_s)
        curr_idx += total_cycles

    X_train_final = np.concatenate(X_train_list, axis=0)
    y_train_final = np.concatenate(y_train_list, axis=0)
    X_test_final, y_test_final = apply_window(test_features, test_y_interp, window_length)

    # Final split
    train_X, vali_X, train_y, vali_y = train_test_split(X_train_final, y_train_final,
                                                        test_size=validation_rate, random_state=42)

    print(f"Train Shape: {train_X.shape}, Test Shape: {X_test_final.shape}")
    return train_X, train_y, vali_X, vali_y, X_test_final, y_test_final


def da_get_xjtu_data_(**kwargs):
    """Wrapper for Domain Adaptation data loading."""
    # Logic remains same as original but calls the cleaned get_xjtu_data_
    res = get_xjtu_data_(**kwargs)
    # Mapping source to target for DA structure
    return res[0], res[0], res[2], res[2], res[1], res[3], res[4], res[5]


if __name__ == '__main__':
    # Usage example
    train_X, train_y, vali_X, vali_y, test_X, test_y = get_xjtu_data_(
        pre_process_type="Vibration",
        root_dir='./XJTU/XJTU-SY_Bearing_Datasets/35Hz12kN',
        train_bearing_data_set=["Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"],
        test_bearing_data_set=["Bearing1_1"],
        STFT_window_len=256, STFT_overlap_num=32, window_length=32, validation_rate=0.1
    )

    # DataLoader setup
    train_loader = DataLoader(XJTUData(train_X, train_y), batch_size=32, shuffle=True)
    test_loader = DataLoader(XJTUData(test_X, test_y), batch_size=32, shuffle=False)