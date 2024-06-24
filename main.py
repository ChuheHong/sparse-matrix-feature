import os
import sys
import pandas as pd
import scipy.io
import scipy.sparse
import numpy as np

# 读取命令行参数（.mtx文件的路径）
file_path = sys.argv[1]

# 读取 .mtx 文件
matrix = scipy.io.mmread(file_path)

# 确保矩阵是CSR格式
if not isinstance(matrix, scipy.sparse.csr_matrix):
    matrix = matrix.tocsr()

# 提取特征
nrow, ncol = matrix.shape
nnz = matrix.nnz

# 行和列的非零元素计数
row_counts = matrix.getnnz(axis=1)
col_counts = matrix.getnnz(axis=0)

max_nnz_per_row = np.max(row_counts)
min_nnz_per_row = np.min(row_counts)
avg_nnz_per_row = np.mean(row_counts)
std_nnz_per_row = np.std(row_counts)

max_nnz_per_col = np.max(col_counts)
min_nnz_per_col = np.min(col_counts)
avg_nnz_per_col = np.mean(col_counts)
std_nnz_per_col = np.std(col_counts)

density = nnz / (nrow * ncol)

# 非零元素计数的方差和标准差
nnz_variance = np.var(row_counts)
nnz_std = np.std(row_counts)

# 计算对角线数量和对角线比例（DIA格式）
row_indices, col_indices = matrix.nonzero()
diagonals = np.unique(col_indices - row_indices)
dia_num = len(diagonals)
dia_ratio = dia_num / min(nrow, ncol)

# 变异系数
CV = std_nnz_per_row / avg_nnz_per_row if avg_nnz_per_row != 0 else float('inf')

# 打印所有特征
# print(f"nrow: {nrow}")
# print(f"ncol: {ncol}")
# print(f"nnz: {nnz}")
# print(f"max_nnz_per_row: {max_nnz_per_row}")
# print(f"min_nnz_per_row: {min_nnz_per_row}")
# print(f"avg_nnz_per_row: {avg_nnz_per_row:.6f}")
# print(f"std_nnz_per_row: {std_nnz_per_row:.6f}")
# print(f"density: {density:.6f}")
# print(f"max_nnz_per_col: {max_nnz_per_col}")
# print(f"min_nnz_per_col: {min_nnz_per_col}")
# print(f"avg_nnz_per_col: {avg_nnz_per_col:.6f}")
# print(f"std_nnz_per_col: {std_nnz_per_col:.6f}")
# print(f"nnz_variance: {nnz_variance:.6f}")
# print(f"nnz_std: {nnz_std:.6f}")
# print(f"dia_num: {dia_num}")
# print(f"dia_ratio: {dia_ratio:.6f}")
# print(f"CV: {CV:.6f}")

# 获取文件名（不包括扩展名）
filename = os.path.splitext(os.path.basename(file_path))[0]

# 创建一个字典，其中包含所有的特征
features = {
    "matrix_name": filename,
    "nrow": nrow,
    "ncol": ncol,
    "nnz": nnz,
    "max_nnz_per_row": max_nnz_per_row,
    "min_nnz_per_row": min_nnz_per_row,
    "avg_nnz_per_row": avg_nnz_per_row,
    "std_nnz_per_row": std_nnz_per_row,
    "density": density,
    "max_nnz_per_col": max_nnz_per_col,
    "min_nnz_per_col": min_nnz_per_col,
    "avg_nnz_per_col": avg_nnz_per_col,
    "std_nnz_per_col": std_nnz_per_col,
    "nnz_variance": nnz_variance,
    "nnz_std": nnz_std,
    "dia_num": dia_num,
    "dia_ratio": dia_ratio,
    "CV": CV
}

# 将字典转换为DataFrame
df = pd.DataFrame([features])

# 将DataFrame写入CSV文件，如果文件已存在，则在其后追加数据
df.to_csv('features.csv', mode='a', header=False, index=False)