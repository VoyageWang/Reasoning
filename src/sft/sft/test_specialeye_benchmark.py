#!/usr/bin/env python3
"""
SpecialEye Benchmark 评估脚本 - 使用 ILVR 模型

基于 RoboRefer/Evaluation/test_benchmark.py 改编，适配 Qwen2.5-VL ILVR 模型
"""
import argparse
import json
import os
import time
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def load_ilvr_model(model_dir: str, base_model_dir: str = None):
    """
    加载 ILVR 模型和 processor

    Args:
        model_dir: ILVR 模型目录
        base_model_dir: 备用的基础模型目录（用于加载 processor）

    Returns:
        model, processor
    """
    print(f"📦 加载模型: {model_dir}")

    # 加载 processor
    processor = None

    # 尝试从多个位置加载 processor
    try:
        processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True
        )
        print("✓ Processor 从模型目录加载成功")
    except Exception as e1:
        print(f"⚠️  从模型目录加载失败: {e1}")

        if base_model_dir and os.path.exists(base_model_dir):
            try:
                processor = AutoProcessor.from_pretrained(
                    base_model_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )
                print(f"✓ Processor 从基础模型加载成功: {base_model_dir}")
            except Exception as e2:
                print(f"⚠️  从基础模型加载失败: {e2}")

        if processor is None:
            common_paths = [
                "/ytech_m2v_hdd/wangyuji/model/Qwen/Qwen2.5-VL-7B-Instruct",
                "/ytech_m2v_hdd/wangyuji/ckpt/ilvr_final/first",
            ]
            for path in common_paths:
                if os.path.exists(path):
                    try:
                        processor = AutoProcessor.from_pretrained(
                            path,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                        print(f"✓ Processor 从备用路径加载成功: {path}")
                        break
                    except Exception:
                        continue

    if processor is None:
        raise RuntimeError("❌ 无法加载 processor，请检查模型目录或网络连接")

    # 加载模型
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            local_files_only=True
        )
    except Exception as e:
        print(f"⚠️  Flash Attention 2 加载失败，使用 eager 模式...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=True
        )

    model.eval()
    print("✅ 模型加载完成")

    return model, processor


def query_ilvr_model(model, processor, image_paths, prompt, max_new_tokens=2048, temperature=0.0):
    """
    使用 ILVR 模型进行推理

    Args:
        model: ILVR 模型
        processor: Processor
        image_paths: 图片路径列表
        prompt: 问题文本
        max_new_tokens: 最大生成token数
        temperature: 温度参数

    Returns:
        dict: {"raw_output": 原始输出, "cleaned_output": 清理后的输出}
    """
    try:
        # 加载图片
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]

        # 构建对话内容
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})

        conversations = [
            {"role": "user", "content": content}
        ]

        # 应用 chat template
        prompt_text = processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True
        )

        # 处理输入
        inputs = processor(
            text=[prompt_text],
            images=images,
            return_tensors="pt",
            padding=True
        )

        # 移动到设备
        inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # 生成配置
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "do_sample": (temperature > 0)
        }

        # 生成
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # 解码（保留特殊tokens）
        raw_output = processor.batch_decode(
            output_ids,
            skip_special_tokens=False
        )[0].strip()

        # 提取 assistant 部分
        import re
        match = re.search(
            r"<\|im_start\|>\s*assistant\s*(.*?)(?:<\|im_end\|>|$)",
            raw_output,
            flags=re.S | re.I
        )

        if match:
            assistant_content = match.group(1)
        else:
            assistant_content = raw_output

        # 清理特殊 tokens
        special_tokens = [
            "<|im_end|>", "<|endoftext|>", "<|eot_id|>",
            "<|latent_start|>", "<|latent_end|>", "<|latent_pad|>",
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"
        ]
        cleaned_output = assistant_content
        for token in special_tokens:
            cleaned_output = cleaned_output.replace(token, "")

        cleaned_output = cleaned_output.strip()

        print(f"Model response: {cleaned_output[:200]}...")

        return {
            "raw_output": raw_output,
            "cleaned_output": cleaned_output
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(1)
        return {
            "raw_output": f'Failed: Query Error - {e}',
            "cleaned_output": 'Failed: Query Error'
        }


def get_prompt(model_name, object_name, prompt, suffix):
    """
    构建 prompt（根据任务类型调整）

    对于 SpecialEye spatial reasoning，要求在 <answer> 标签中输出 JSON 格式
    """
    # 添加格式要求
    formatted_prompt = f"""{prompt}

Output the point coordinates in JSON format inside <answer></answer> tags.
For example:
<answer>
[
{{"point_2d": [x, y], "label": "point_1"}}
]
</answer>"""

#[
# {{"point_2d": [x, y], "label": "point_1"}}
# ]

# [(x, y)]
    return formatted_prompt


def eval_task(task_name, model_dir, base_model_dir, output_save_folder, max_new_tokens=2048):
    """
    评估指定任务

    Args:
        task_name: 任务名称 (例如: "Location", "Placement", "Unseen")
        model_dir: ILVR 模型目录
        base_model_dir: 基础模型目录（用于加载processor）
        output_save_folder: 输出文件夹
        max_new_tokens: 最大生成token数
    """
    benchmark_question_file = f"/ytech_m2v_hdd/wangyuji/data/JingkunAn/RefSpatial-Bench/{task_name}"

    # 加载模型（只加载一次）
    model, processor = load_ilvr_model(model_dir, base_model_dir)

    # 加载问题
    question_file = f"{benchmark_question_file}/question.json"
    if not os.path.exists(question_file):
        print(f"❌ 问题文件不存在: {question_file}")
        return

    with open(question_file, "r") as f:
        questions = json.load(f)

    print(f"📊 加载了 {len(questions)} 个问题")

    # 创建输出路径
    model_name_short = os.path.basename(model_dir)
    output_path = f'{output_save_folder}/{model_name_short}/{task_name}.jsonl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f'⚠️  {output_path} 已存在，跳过')
        return

    print(f"💾 输出文件: {output_path}")

    # 推理并保存结果
    with open(output_path, "w") as ans_file:
        for idx, question in enumerate(questions):
            # 构建图片路径
            image_paths = [f"{benchmark_question_file}/{question['rgb_path']}"]

            # 检查图片是否存在
            if not os.path.exists(image_paths[0]):
                print(f"⚠️  图片不存在: {image_paths[0]}")
                continue

            # 构建 prompt
            instruction = get_prompt(
                model_dir,
                question["object"],
                question["prompt"],
                question["suffix"]
            )

            print(f"\n[{idx+1}/{len(questions)}] Processing question {question['id']}")
            print(f"Prompt: {instruction[:100]}...")

            # 生成答案
            model_output = query_ilvr_model(
                model,
                processor,
                image_paths,
                instruction,
                max_new_tokens=max_new_tokens,
                temperature=0.0
            )

            # 保存结果（包含原始输出和清理后的输出）
            result = {
                "question_id": question["id"],
                "prompt": question["prompt"],
                "object_name": question["object"],
                "suffix": question["suffix"],
                "instruction": instruction,
                "text": model_output["cleaned_output"],  # 清理后的输出
                "raw_output": model_output["raw_output"],  # 原始输出（包含特殊tokens）
                "model_id": model_dir,
                "rgb_path": question["rgb_path"],
                "mask_path": question["mask_path"],
                "category": question["category"],
                "step": question["step"]
            }

            ans_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            ans_file.flush()

    print(f"\n✅ 任务 {task_name} 完成！")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpecialEye Benchmark 评估 - ILVR 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估单个任务
  python test_specialeye_benchmark.py \\
    --model_dir /ytech_m2v_hdd/wangyuji/ckpt/ilvr_depth/step_2000 \\
    --task_name Location

  # 评估所有任务
  python test_specialeye_benchmark.py \\
    --model_dir /ytech_m2v_hdd/wangyuji/ckpt/ilvr_depth/step_2000 \\
    --task_name all

  # 自定义输出目录和生成长度
  python test_specialeye_benchmark.py \\
    --model_dir /ytech_m2v_hdd/wangyuji/ckpt/ilvr_final/covt_ilvr \\
    --task_name Location Placement \\
    --output_folder ./benchmark_results \\
    --max_new_tokens 4096
        """
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="/ytech_m2v_hdd/wangyuji/ckpt/ilvr_final/covt_ilvr",
        help="ILVR 模型目录路径"
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="/ytech_m2v_hdd/wangyuji/model/Qwen/Qwen2.5-VL-7B-Instruct",
        help="基础模型目录（用于加载 processor）"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        nargs="+",
        default=['Location'],
        help="任务名称，可选: Location, Placement, Unseen, all"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default='./benchmark_outputs',
        help="输出文件夹路径"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="最大生成 token 数"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("="*60)
    print("🚀 SpecialEye Benchmark 评估 - ILVR 模型")
    print("="*60)
    print(f"模型: {args.model_dir}")
    print(f"输出: {args.output_folder}")
    print(f"最大tokens: {args.max_new_tokens}")
    print("="*60)

    # 处理任务列表
    if args.task_name == ['all']:
        subtasks = ['Location', 'Placement', "Unseen"]
    else:
        subtasks = args.task_name

    # 评估每个任务
    for task_name in subtasks:
        print(f'\n{"="*60}')
        print(f'📋 处理任务: {task_name}')
        print(f'{"="*60}')

        eval_task(
            task_name=task_name,
            model_dir=args.model_dir,
            base_model_dir=args.base_model_dir,
            output_save_folder=args.output_folder,
            max_new_tokens=args.max_new_tokens
        )

    print(f'\n{"="*60}')
    print("🎉 所有任务完成！")
    print(f'{"="*60}')
