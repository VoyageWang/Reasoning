#!/usr/bin/env python3
"""
转换 SpecialEye spatial 数据格式为 COMT 训练格式
"""
import json
import argparse
import os
from typing import List, Dict, Any


def extract_relative_path(full_path: str) -> str:
    """
    从完整路径中提取相对路径（只保留最后两级）
    例如: /ytech_m2v_hdd/.../image/0014abacca6a504b.jpg -> image/0014abacca6a504b.jpg
    """
    parts = full_path.split(os.sep)
    # 获取最后两部分: image/filename 或 depth/filename
    if len(parts) >= 2:
        return os.path.join(parts[-2], parts[-1])
    return full_path


def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    转换单个样本

    输入格式:
    {
        "id": "...",
        "image_path": "...",
        "depth_path": "...",
        "question": "...",
        "answer": "...",
        "cot": "O_pre:\n...\n\nO_post:\n..."
    }

    输出格式 (COMT):
    {
        "text_input": question,
        "image_input": [image_path],
        "sequence_plan": [
            {"type": "text", "content": "<think>O_pre</think>"},
            {"type": "latent", "helper_image": depth_path},
            {"type": "text", "content": "<think>O_post</think>"},
            {"type": "text", "content": "<answer>answer</answer>"}
        ],
        "original_final_answer": answer
    }
    """
    # 提取 O_pre 和 O_post
    cot = sample.get("cot", "")

    # 分割 O_pre 和 O_post
    parts = cot.split("\n\nO_post:\n")

    if len(parts) == 2:
        o_pre = parts[0].replace("O_pre:\n", "").strip()
        o_post = parts[1].strip()
    else:
        # 如果格式不对,尝试其他分割方式
        parts = cot.split("O_post:")
        if len(parts) == 2:
            o_pre = parts[0].replace("O_pre:", "").strip()
            o_post = parts[1].strip()
        else:
            # 默认处理
            o_pre = cot.strip()
            o_post = ""

    # 提取相对路径
    image_path = extract_relative_path(sample["image_path"])
    depth_path = extract_relative_path(sample["depth_path"])

    # 构建 sequence_plan
    sequence_plan = [
        {
            "type": "text",
            "content": f"<think>{o_pre}</think>"
        },
        {
            "type": "latent",
            "helper_image": depth_path
        },
        {
            "type": "text",
            "content": f"<think>{o_post}</think>"
        },
        {
            "type": "text",
            "content": f"<answer>{sample['answer']}</answer>"
        }
    ]

    # 构建输出样本
    converted = {
        "text_input": sample["question"],
        "image_input": [image_path],
        "sequence_plan": sequence_plan,
        "original_final_answer": sample["answer"]
    }

    return converted


def main():
    parser = argparse.ArgumentParser(
        description="转换 SpecialEye spatial 数据为 COMT 格式"
    )
    parser.add_argument(
        "--input_choice",
        type=str,
        required=True,
        help="输入的 choice 数据文件 (spatial_data_cot_choice.json.tmp)"
    )
    parser.add_argument(
        "--input_qa",
        type=str,
        required=True,
        help="输入的 QA 数据文件 (spatial_data_cot_qa.json.tmp)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出的 JSONL 文件路径"
    )

    args = parser.parse_args()

    print(f"读取 choice 数据: {args.input_choice}")
    choice_data = []
    with open(args.input_choice, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                choice_data.append(json.loads(line))

    print(f"读取 QA 数据: {args.input_qa}")
    qa_data = []
    with open(args.input_qa, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_data.append(json.loads(line))

    # 合并数据
    all_data = choice_data + qa_data
    print(f"总样本数: {len(all_data)}")

    # 转换每个样本
    converted_samples = []
    for i, sample in enumerate(all_data):
        try:
            converted = convert_sample(sample)
            converted_samples.append(converted)

            if (i + 1) % 100 == 0:
                print(f"已转换: {i + 1}/{len(all_data)}")
        except Exception as e:
            print(f"警告: 样本 {i} (id={sample.get('id', 'unknown')}) 转换失败: {e}")
            continue

    # 写入 JSONL 文件
    print(f"\n写入输出文件: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n✅ 转换完成!")
    print(f"输入样本: {len(all_data)}")
    print(f"成功转换: {len(converted_samples)}")
    print(f"输出文件: {args.output}")

    # 显示第一个样本作为示例
    if converted_samples:
        print("\n第一个转换样本示例:")
        print(json.dumps(converted_samples[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
