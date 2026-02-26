#!/usr/bin/env python3
"""
将模型转换为 HuggingFace safetensors 格式

支持两种输入:
1. DeepSpeed checkpoint (包含 zero_to_fp32.py)
2. 已有 pytorch_model.bin 的目录
"""
import os
import sys
import argparse
import subprocess
import shutil
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def is_deepspeed_checkpoint(checkpoint_dir: str) -> bool:
    """
    检查是否为 DeepSpeed checkpoint

    Args:
        checkpoint_dir: checkpoint 目录路径

    Returns:
        True if DeepSpeed checkpoint, False otherwise
    """
    zero_to_fp32_script = os.path.join(checkpoint_dir, "zero_to_fp32.py")
    global_step_dirs = [d for d in os.listdir(checkpoint_dir)
                       if d.startswith("global_step") and os.path.isdir(os.path.join(checkpoint_dir, d))]

    return os.path.exists(zero_to_fp32_script) or len(global_step_dirs) > 0


def convert_deepspeed_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    remove_temp: bool = True
):
    """
    将 DeepSpeed checkpoint 转换为 pytorch_model 格式

    Args:
        checkpoint_dir: DeepSpeed checkpoint 目录
        output_dir: 输出目录
        remove_temp: 是否删除临时文件
    """
    print("\n🔧 检测到 DeepSpeed checkpoint，开始转换...")

    zero_to_fp32_script = os.path.join(checkpoint_dir, "zero_to_fp32.py")
    if not os.path.exists(zero_to_fp32_script):
        raise FileNotFoundError(f"未找到 zero_to_fp32.py: {zero_to_fp32_script}")

    # Step 1: 使用 zero_to_fp32.py 转换
    print("\n步骤 1/4: 转换 DeepSpeed checkpoint 为 pytorch_model...")
    temp_model_path = os.path.join(output_dir, "pytorch_model.bin")

    cmd = [sys.executable, zero_to_fp32_script, checkpoint_dir, temp_model_path]
    print(f"执行: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ 转换失败！")
        print(f"错误: {result.stderr}")
        raise RuntimeError("DeepSpeed checkpoint 转换失败")

    print("✓ 成功生成 pytorch_model")

    # 如果 zero_to_fp32.py 创建了目录，移动文件
    if os.path.isdir(temp_model_path):
        print("检测到目录格式，正在移动文件...")
        for item in os.listdir(temp_model_path):
            src = os.path.join(temp_model_path, item)
            dst = os.path.join(output_dir, item)
            shutil.move(src, dst)
            print(f"  移动: {item}")
        os.rmdir(temp_model_path)

    # Step 2: 复制配置文件
    print("\n步骤 2/4: 复制配置文件和 tokenizer...")
    files_to_copy = [
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json"
    ]

    for filename in files_to_copy:
        src = os.path.join(checkpoint_dir, filename)
        if os.path.exists(src):
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            print(f"  ✓ {filename}")


def convert_to_safetensors(
    model_dir: str,
    output_dir: str = None,
    remove_pytorch: bool = True
):
    """
    加载模型并保存为 safetensors 格式

    Args:
        model_dir: 模型目录（包含 pytorch_model 或已是输出目录）
        output_dir: 输出目录（如果为 None，则在原地转换）
        remove_pytorch: 是否删除 pytorch_model 文件
    """
    if output_dir is None:
        output_dir = model_dir

    print("\n步骤 3/4: 加载模型并保存为 safetensors...")
    print("正在加载模型（这可能需要几分钟）...")

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            output_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ 模型加载成功")

        print("\n保存为 safetensors 格式...")
        model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        print("✓ 模型保存成功")

        # 保存 processor
        try:
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
            processor.save_pretrained(output_dir)
            print("✓ Processor 已保存")
        except Exception as e:
            print(f"⚠️  Processor 保存失败: {e}")

    except Exception as e:
        print(f"❌ 模型加载/保存失败: {e}")
        raise

    # Step 4: 清理临时文件
    if remove_pytorch:
        print("\n步骤 4/4: 清理 pytorch_model 文件...")
        for f in os.listdir(output_dir):
            if f.startswith("pytorch_model"):
                fpath = os.path.join(output_dir, f)
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    print(f"  🗑️  删除: {f}")
    else:
        print("\n步骤 4/4: 跳过清理（保留 pytorch_model 文件）")

    # 显示最终文件
    print("\n✅ 转换完成！")
    print(f"📁 输出目录: {output_dir}")
    print("\n最终文件列表:")

    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / (1024**3)
            icon = "🔒" if f.endswith(".safetensors") else "📄"
            print(f"  {icon} {f} ({size:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(
        description="将模型转换为 safetensors 格式（自动检测输入类型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # DeepSpeed checkpoint 转换
  python convert_to_safetensors.py \\
    --checkpoint_dir /path/to/checkpoint-810 \\
    --output_dir /path/to/output

  # 原地转换已有的 pytorch_model
  python convert_to_safetensors.py \\
    --checkpoint_dir /path/to/model_with_pytorch_bin

  # 保留 pytorch_model 文件
  python convert_to_safetensors.py \\
    --checkpoint_dir /path/to/checkpoint \\
    --output_dir /path/to/output \\
    --keep_pytorch
        """
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint 目录路径（DeepSpeed 或 pytorch_model）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录路径（默认：原地转换）"
    )
    parser.add_argument(
        "--keep_pytorch",
        action="store_true",
        help="保留 pytorch_model 文件"
    )

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.exists(args.checkpoint_dir):
        print(f"❌ 错误: Checkpoint 目录不存在: {args.checkpoint_dir}")
        sys.exit(1)

    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = args.checkpoint_dir
        print(f"ℹ️  未指定输出目录，将原地转换: {args.output_dir}")
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 开始模型转换")
    print(f"{'='*60}")
    print(f"输入: {args.checkpoint_dir}")
    print(f"输出: {args.output_dir}")
    print(f"{'='*60}")

    # 自动检测并转换
    if is_deepspeed_checkpoint(args.checkpoint_dir):
        print("\n📦 检测类型: DeepSpeed Checkpoint")
        convert_deepspeed_checkpoint(
            args.checkpoint_dir,
            args.output_dir,
            remove_temp=True
        )
    else:
        print("\n📦 检测类型: PyTorch Model (已有 pytorch_model)")

    # 转换为 safetensors
    convert_to_safetensors(
        args.checkpoint_dir,
        args.output_dir,
        remove_pytorch=not args.keep_pytorch
    )

    print(f"\n{'='*60}")
    print(f"🎉 全部完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
