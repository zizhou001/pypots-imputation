from datetime import datetime

import numpy as np
import pandas as pd
import torch
from pypots.utils.metrics import calc_mae, calc_rmse
from sklearn.preprocessing import StandardScaler

import CustomDataset


# 读取数据
def load_data(file_path, columns=None):
    """
    Load data from a CSV file and optionally select specific columns.

    Parameters:
    - file_path: str, path to the CSV file
    - columns: list of str, optional, list of column names to select from the CSV file

    Returns:
    - pd.DataFrame containing the selected columns
    """
    # 读取 CSV 文件到 DataFrame
    df = pd.read_csv(file_path)

    # 如果指定了要提取的列，则选择这些列
    if columns is not None:
        # 确保列存在于 DataFrame 中
        invalid_columns = [col for col in columns if col not in df.columns]
        if invalid_columns:
            raise ValueError(f"Columns not found in the DataFrame: {', '.join(invalid_columns)}")
        df = df[columns]

    return df


# 标准化数据
def standardize_data(data):
    """
    标准化数据，并保持原始列名。

    参数:
    data (pd.DataFrame or np.ndarray): 输入数据，可以是 pandas DataFrame 或 numpy ndarray。

    返回:
    pd.DataFrame: 标准化后的数据，保留原始列名（如果输入是 DataFrame）。
    StandardScaler: 训练好的标准化器。
    """
    # 检查输入数据类型
    if isinstance(data, pd.DataFrame):
        columns = data.columns  # 保存列名
        data_values = data.values  # 获取数据的 ndarray 表示
    elif isinstance(data, np.ndarray):
        columns = None  # 如果输入是 ndarray，则没有列名
        data_values = data
    else:
        raise ValueError("Input data must be a pandas DataFrame or numpy ndarray")

    # 如果数据是一维的，将其转换为二维
    if data_values.ndim == 1:
        data_values = data_values.reshape(-1, 1)

    # 标准化数据
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_values)

    # 如果输入是 DataFrame，则恢复为 DataFrame，保留列名
    if columns is not None:
        data_standardized = pd.DataFrame(data_standardized, columns=columns)

    return data_standardized, scaler


# 使用训练好的模型进行插补
def impute_data(model, dataloader, scaler):
    imputed_data = model.predict(dataloader)
    imputed_data = scaler.inverse_transform(imputed_data)
    return imputed_data


# 计算RMSE和MAE
def calculate_errors(true_data, imputed_data, mask):
    # print(imputed_data.shape, true_data.shape, mask.shape)

    if imputed_data.shape[1] == 1:
        imputed_data = imputed_data.squeeze(1)

    middle_time_step = imputed_data.shape[1] // 2
    selected_imputed_data = imputed_data[:, middle_time_step, :]

    # 裁剪 true_data 和 mask 的第一个维度
    num_rows = selected_imputed_data.shape[0]
    true_data_cropped = true_data.iloc[:num_rows, :]
    mask_cropped = mask[:num_rows, :]

    # 将 DataFrame 转换为 NumPy 数组以便进行掩码操作
    selected_imputed_data_np = selected_imputed_data
    true_data_cropped_np = true_data_cropped.to_numpy()
    mask_cropped_np = mask_cropped.cpu().numpy()

    # 应用掩码
    masked_imputed_data = selected_imputed_data_np * mask_cropped_np
    masked_true_data = true_data_cropped_np * mask_cropped_np

    mae = calc_mae(masked_true_data, masked_imputed_data)
    rmse_value = calc_rmse(masked_true_data, masked_imputed_data)

    return rmse_value, mae


# 将结果输出到文件
def write_results_to_file(results, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        # 添加日期和时间信息
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Date and Time: {date_time}\n")

        for model_name, metrics in results.items():
            f.write(f"\t{model_name} - MR: {metrics['MR']:.2f} - MML: {metrics['MML']:.2f} - RMSE: {metrics['RMSE']:.3f} - MAE: {metrics['MAE']:.3f}\n")


# 划分数据集
def split_data(data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, "The sum of ratios must be equal to 1."

    n_samples = len(data)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    test_size = n_samples - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


# 生成具有缺失值的数据
def generate_missing_data(df, column_names, missing_rate=0.1, max_missing_length=24, missing_mode='continuous'):
    df_copy = df.copy()

    # 确保 column_names 是列表并且在 DataFrame 中有效
    if not isinstance(column_names, list):
        raise ValueError("column_names should be a list of column names")

    # 确保所有列名都存在于 DataFrame 中
    invalid_columns = [col for col in column_names if col not in df_copy.columns]
    if invalid_columns:
        raise ValueError(f"Invalid column names: {invalid_columns}")

    num_rows = len(df_copy)
    columns = df_copy[column_names].values

    # 初始化掩码矩阵
    mask = np.ones(columns.shape, dtype=np.float32)

    if missing_mode == 'random':
        # 随机缺失
        num_missing = int(num_rows * missing_rate)
        missing_indices = np.random.choice(num_rows, num_missing, replace=False)
        mask[missing_indices] = 0
    elif missing_mode == 'continuous':
        # 连续长序列缺失
        num_missing = int(num_rows * missing_rate)
        if max_missing_length > num_missing:
            max_missing_length = num_missing
        for column_index in range(columns.shape[1]):
            num_segments = max(int(num_missing / max_missing_length), 1)
            centering = (num_segments == 1)
            for _ in range(num_segments):
                if centering:
                    # Center the segment
                    middle_index = num_rows // 2
                    start_index = max(0, middle_index - max_missing_length // 2)
                    end_index = min(start_index + max_missing_length, num_rows)

                    # Adjust start index to fit within bounds
                    if end_index - start_index < max_missing_length:
                        start_index = max(0, end_index - max_missing_length)

                else:
                    # Randomly select start index
                    start_index = np.random.randint(0, num_rows - max_missing_length + 1)
                    end_index = start_index + max_missing_length

                # Ensure indices are integers
                start_index = int(start_index)
                end_index = int(end_index)

                # Ensure indices are within bounds and do not exceed the number of rows
                start_index = max(0, start_index)
                end_index = min(num_rows, end_index)

                # Apply the mask
                mask[start_index:end_index, column_index] = 0

                # Ensure mask length reaches expected proportion
                if np.sum(mask[:, column_index] == 0) >= num_missing:
                    break
    else:
        raise ValueError("Invalid missing_mode. Choose between 'random' and 'continuous'")

    # 将掩码应用到数据上
    masked_data = np.where(mask == 0, np.nan, columns)

    # 转换掩码矩阵为 tensor
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    return masked_data, mask_tensor


def reshape_data(data, seq_length):
    num_samples = len(data) - seq_length + 1
    n_features = data.shape[1]

    # 创建一个新的数组来存储重塑后的数据
    reshaped_data = np.zeros((num_samples, seq_length, n_features))

    # 填充重塑后的数据
    for i in range(num_samples):
        reshaped_data[i] = data[i:i + seq_length]

    return reshaped_data


# 比较不同模型的插补效果
def compare_models(data, models, parameters, epochs=100, batch_size=32, missing_mode='continuous', missing_rate=0.2,
                   device=None, n_features=2, n_steps=64, patience=5, max_missing_length=25):
    results = {}

    for model_name, model_class in models.items():
        print(f"Processing model: {model_name}")

        # 标准化数据
        data_standardized, scaler = standardize_data(data)

        # 分割数据
        train_data, val_data, test_data = split_data(data=data_standardized)

        # 生成具有缺失值的数据
        train_data_missing, train_mask = generate_missing_data(train_data, column_names=['windSpeed2m', 'windSpeed10m'],
                                                               missing_rate=missing_rate, missing_mode=missing_mode,
                                                               max_missing_length=max_missing_length)
        test_data_missing, test_mask = generate_missing_data(test_data, column_names=['windSpeed2m', 'windSpeed10m'],
                                                             missing_rate=missing_rate, missing_mode=missing_mode,
                                                             max_missing_length=max_missing_length)

        # 将数据重塑为 (num_samples, seq_length, n_features)
        train_data_reshaped = reshape_data(train_data_missing, n_steps)
        test_data_reshaped = reshape_data(test_data_missing, n_steps)

        train_dict = CustomDataset.create_dict(train_data_reshaped, batch_size=batch_size, masks=train_mask)
        test_dict = CustomDataset.create_dict(test_data_reshaped, batch_size=batch_size, masks=test_mask)

        # 获取模型特定的参数
        model_params = parameters.get(model_name, {})

        # 训练模型
        model = model_class(**model_params, epochs=epochs, batch_size=batch_size, patience=patience, n_steps=n_steps,
                            n_features=n_features)
        model.fit(train_dict)

        # 使用模型插补
        imputed_data = model.impute(test_dict)

        # 计算RMSE和MAE
        rmse, mae = calculate_errors(test_data, imputed_data, test_mask)

        results[model_name] = {'MR': missing_rate, 'MML': max_missing_length, 'RMSE': rmse, 'MAE': mae}

    return results



if __name__ == "__main__":
    pass
