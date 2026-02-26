#!/usr/bin/env python3
"""
数据集混合脚本

功能：
1. 从两个数据集中按比例采样
2. 合并成新的数据集
3. 保持parquet格式

使用方法：
python merge_datasets.py --thyme_ratio 0.7 --covt_ratio 0.3 --output_dir /path/to/output
"""

import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import random
from tqdm import tqdm


def load_dataset_files(dataset_dir):
    """加载数据集中的所有parquet文件"""
    dataset_path = Path(dataset_dir)
    parquet_files = sorted(dataset_path.glob("train-*.parquet"))
    return parquet_files


def count_total_rows(parquet_files):
    """统计数据集总行数"""
    total_rows = 0
    for file in tqdm(parquet_files, desc="统计行数"):
        table = pq.read_table(file)
        total_rows += len(table)
    return total_rows


def sample_from_dataset(parquet_files, sample_size, seed=42):
    """
    从数据集中随机采样指定数量的样本（分批处理避免内存溢出）

    Args:
        parquet_files: parquet文件列表
        sample_size: 采样数量
        seed: 随机种子

    Returns:
        采样后的PyArrow Table
    """
    random.seed(seed)

    # 第一步：统计每个文件的行数
    print(f"正在统计 {len(parquet_files)} 个文件的行数...")
    file_row_counts = []
    total_rows = 0
    for file in tqdm(parquet_files, desc="统计行数"):
        table = pq.read_table(file)
        row_count = len(table)
        file_row_counts.append(row_count)
        total_rows += row_count

    print(f"总行数: {total_rows}, 采样数: {sample_size}")

    if sample_size >= total_rows:
        print("采样数大于等于总行数，返回全部数据")
        # 分批读取并合并
        tables = []
        for file in tqdm(parquet_files, desc="读取文件"):
            table = pq.read_table(file)
            tables.append(table)
        return pa.concat_tables(tables)

    # 第二步：生成全局随机索引
    print("生成随机索引...")
    global_indices = sorted(random.sample(range(total_rows), sample_size))

    # 第三步：将全局索引映射到每个文件
    print("映射索引到文件...")
    file_indices = [[] for _ in range(len(parquet_files))]
    cumulative_rows = 0
    global_idx_ptr = 0

    for file_idx, row_count in enumerate(file_row_counts):
        file_start = cumulative_rows
        file_end = cumulative_rows + row_count

        # 找到属于当前文件的所有索引
        while global_idx_ptr < len(global_indices) and global_indices[global_idx_ptr] < file_end:
            global_idx = global_indices[global_idx_ptr]
            if global_idx >= file_start:
                # 转换为文件内的局部索引
                local_idx = global_idx - file_start
                file_indices[file_idx].append(local_idx)
            global_idx_ptr += 1

        cumulative_rows += row_count

    # 第四步：从每个文件中采样并合并
    print("从各文件采样...")
    sampled_tables = []
    for file_idx, file in enumerate(tqdm(parquet_files, desc="采样文件")):
        if len(file_indices[file_idx]) > 0:
            table = pq.read_table(file)
            sampled = table.take(file_indices[file_idx])
            sampled_tables.append(sampled)

    # 合并采样结果
    print("合并采样结果...")
    return pa.concat_tables(sampled_tables)


def merge_and_save(thyme_table, covt_table, output_dir, rows_per_file=1000):
    """
    合并两个表并保存为parquet文件

    Args:
        thyme_table: Thyme-RL采样表
        covt_table: covt-rl采样表
        output_dir: 输出目录
        rows_per_file: 每个文件的行数
    """
    # 合并表
    print("合并数据集...")
    merged_table = pa.concat_tables([thyme_table, covt_table])
    total_rows = len(merged_table)

    print(f"合并后总行数: {total_rows}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 计算需要的文件数
    num_files = (total_rows + rows_per_file - 1) // rows_per_file

    print(f"将保存为 {num_files} 个文件...")

    # 分批保存
    for i in tqdm(range(num_files), desc="保存文件"):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, total_rows)

        batch_table = merged_table.slice(start_idx, end_idx - start_idx)

        output_file = output_path / f"train-{i:05d}-of-{num_files:05d}.parquet"
        pq.write_table(batch_table, output_file)

    print(f"✓ 数据集已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="混合两个数据集")
    parser.add_argument("--thyme_dir", type=str,
                        default="<THYME_RL_DATA_PATH>",
                        help="Thyme-RL数据集目录")
    parser.add_argument("--covt_dir", type=str,
                        default="<COVT_RL_DATA_PATH>",
                        help="covt-rl数据集目录")
    parser.add_argument("--output_dir", type=str,
                        default="<OUTPUT_DATA_PATH>",
                        help="输出目录")
    parser.add_argument("--thyme_ratio", type=float, default=0.3,
                        help="Thyme-RL数据集采样比例 (0-1)，1.0表示使用全部数据")
    parser.add_argument("--covt_ratio", type=float, default=1.0,
                        help="covt-rl数据集采样比例 (0-1)，1.0表示使用全部数据")
    parser.add_argument("--total_samples", type=int, default=None,
                        help="总样本数（已废弃，现在直接使用ratio控制）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--rows_per_file", type=int, default=1000,
                        help="每个parquet文件的行数")

    args = parser.parse_args()

    print("="*60)
    print("数据集混合工具")
    print("="*60)

    # 加载数据集文件列表
    print("\n1. 加载数据集文件列表...")
    thyme_files = load_dataset_files(args.thyme_dir)
    covt_files = load_dataset_files(args.covt_dir)

    print(f"  Thyme-RL: {len(thyme_files)} 个文件")
    print(f"  covt-rl: {len(covt_files)} 个文件")

    # 统计总行数
    print("\n2. 统计数据集大小...")
    thyme_total = count_total_rows(thyme_files)
    covt_total = count_total_rows(covt_files)

    print(f"  Thyme-RL总行数: {thyme_total}")
    print(f"  covt-rl总行数: {covt_total}")

    # 计算采样数量
    print("\n3. 计算采样数量...")

    # 直接根据ratio计算采样数
    thyme_samples = int(thyme_total * args.thyme_ratio)
    covt_samples = int(covt_total * args.covt_ratio)

    print(f"  Thyme-RL采样数: {thyme_samples} / {thyme_total} ({args.thyme_ratio*100:.1f}%)")
    print(f"  covt-rl采样数: {covt_samples} / {covt_total} ({args.covt_ratio*100:.1f}%)")
    print(f"  混合后总计: {thyme_samples + covt_samples}")

    # 采样
    print("\n4. 从Thyme-RL采样...")
    thyme_table = sample_from_dataset(thyme_files, thyme_samples, args.seed)

    print("\n5. 从covt-rl采样...")
    covt_table = sample_from_dataset(covt_files, covt_samples, args.seed + 1)

    # 合并并保存
    print("\n6. 合并并保存...")
    merge_and_save(thyme_table, covt_table, args.output_dir, args.rows_per_file)

    print("\n" + "="*60)
    print("✓ 完成！")
    print("="*60)


if __name__ == "__main__":
    main()
