#!/bin/bash

echo "Matrix Name, nrow, ncol, nnz, max_nnz_per_row, min_nnz_per_row, avg_nnz_per_row, std_nnz_per_row, density, max_nnz_per_col, min_nnz_per_col, avg_nnz_per_col, std_nnz_per_col, nnz_variance, nnz_std, dia_num, dia_ratio, CV" > features.csv

# 遍历matrix_mtx/目录下的所有.mtx文件
for file in ../matrix_mtx/*.mtx
do
    # 对于每个.mtx文件，运行Python脚本并传递文件名作为参数
    yhrun -N 1 -n 1 -p gpu python3 ./main.py "$file"
done