#!/usr/bin/env python3
"""
ILVR 模型推理 Demo

用法:
    python inference_demo.py --model_dir /path/to/model --image /path/to/image.jpg --question "你的问题"
"""
import os
import argparse
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def load_model_and_processor(model_dir: str, base_model_dir: str = None):
    """
    加载模型和处理器

    Args:
        model_dir: 模型目录路径
        base_model_dir: 备用的基础模型目录（用于加载 processor）

    Returns:
        model, processor
    """
    print(f"📦 加载模型: {model_dir}")

    # 加载 processor - 优先从本地加载
    processor = None

    # 尝试 1: 从模型目录加载（本地优先）
    try:
        processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True
        )
        print("✓ Processor 从模型目录加载成功")
    except Exception as e1:
        print(f"⚠️  从模型目录加载失败: {e1}")

        # 尝试 2: 从备用的基础模型目录加载
        if base_model_dir and os.path.exists(base_model_dir):
            try:
                processor = AutoProcessor.from_pretrained(
                    base_model_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )
                print(f"✓ Processor 从基础模型目录加载成功: {base_model_dir}")
            except Exception as e2:
                print(f"⚠️  从基础模型目录加载失败: {e2}")

        # 尝试 3: 从常见的本地模型路径加载
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

        # 最后尝试：不使用 local_files_only（需要网络）
        if processor is None:
            print("⚠️  所有本地加载尝试失败，尝试从 HuggingFace Hub 下载...")
            try:
                processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    trust_remote_code=True
                )
                print("✓ Processor 从 HuggingFace Hub 下载成功")
            except Exception as e:
                raise RuntimeError(
                    f"❌ 无法加载 processor！\n"
                    f"   请检查:\n"
                    f"   1. 模型目录是否包含 processor 文件\n"
                    f"   2. 基础模型目录是否存在\n"
                    f"   3. 网络连接是否正常\n"
                    f"   错误: {e}"
                )

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
        print(f"⚠️  Flash Attention 2 加载失败 ({e})，使用 eager 模式...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=True
        )

    model.eval()

    # 启用 TF32（如果支持）
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    print("✅ 模型加载完成")
    return model, processor


def inference(
    model,
    processor,
    image_path: str,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0
):
    """
    执行推理

    Args:
        model: 模型
        processor: 处理器
        image_path: 图片路径
        question: 问题
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_p: top-p 采样参数

    Returns:
        生成的文本
    """
    # 加载图片
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    image = Image.open(image_path).convert("RGB")
    print(f"🖼️  图片: {image_path} ({image.size})")
    print(f"❓ 问题: {question}")

    # 构建对话
    content = [
        {"type": "image"},
        {"type": "text", "text": question}
    ]

    conversations = [
        {"role": "user", "content": content}
    ]

    # 应用 chat template
    prompt = processor.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True
    )

    # 处理输入
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    # 移动到模型设备
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # 生成配置
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": (temperature > 0)
    }

    print("\n🤖 正在生成回答...")

    # 生成
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

    # 解码输出
    output_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=False
    )[0].strip()

    return output_text


def extract_answer(output_text: str) -> str:
    """
    从输出中提取答案部分

    Args:
        output_text: 完整输出

    Returns:
        提取的答案
    """
    import re

    # 提取 assistant 部分
    match = re.search(
        r"<\|im_start\|>\s*assistant\s*(.*?)(?:<\|im_end\|>|$)",
        output_text,
        flags=re.S | re.I
    )

    if match:
        assistant_content = match.group(1)
    else:
        # 如果没有找到，尝试其他模式
        match = re.search(
            r"<\|assistant\|>\s*(.*?)(?:<\|im_end\|>|<\|endoftext\|>|$)",
            output_text,
            flags=re.S | re.I
        )
        if match:
            assistant_content = match.group(1)
        else:
            assistant_content = output_text

    # 清理特殊 tokens
    special_tokens = [
        "<|im_end|>", "<|endoftext|>", "<|eot_id|>",
        "<|latent_start|>", "<|latent_end|>", "<|latent_pad|>",
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"
    ]

    for token in special_tokens:
        assistant_content = assistant_content.replace(token, "")

    return assistant_content.strip()


def main():
    parser = argparse.ArgumentParser(
        description="ILVR 模型推理 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张图片推理
  python inference_demo.py \\
    --model_dir /ytech_m2v_hdd/wangyuji/ckpt/ilvr_final/first \\
    --image /path/to/image.jpg \\
    --question "What is in this image?"

  # 调整生成参数
  python inference_demo.py \\
    --model_dir /path/to/model \\
    --image /path/to/image.jpg \\
    --question "Describe this image in detail" \\
    --max_new_tokens 1024 \\
    --temperature 0.7
        """
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="/ytech_m2v_hdd/wangyuji/ckpt/ilvr_final/covt_ilvr",
        help="模型目录路径 (默认: /ytech_m2v_hdd/wangyuji/ckpt/ilvr_final/first)"
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="/ytech_m2v_hdd/wangyuji/model/Qwen/Qwen2.5-VL-7B-Instruct",
        help="基础模型目录，用于加载 processor (默认: /ytech_m2v_hdd/wangyuji/model/Qwen/Qwen2.5-VL-7B-Instruct)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/ytech_m2v_hdd/wangyuji/data/JingkunAn/RefSpatial/2D/image/000d32e2b7d81b8c.jpg",
        help="输入图片路径 (默认: /ytech_m2v_hdd/wangyuji/data/comt/images_comt/creation/12035.png)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Does white stone cross at center appear under smooth stone cross at center? (A) yes (B) no",
        help="问题文本 (默认: What is in this image?)" # Yes, point (0.032, 0.512) is closer to viewer.
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成 token 数 (默认: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="温度参数 (默认: 0.0, 确定性生成)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p 采样参数 (默认: 1.0)"
    )
    parser.add_argument(
        "--show_full",
        action="store_true",
        help="显示完整输出（包括特殊 tokens）"
    )

    args = parser.parse_args()

    print("="*60)
    print("🚀 ILVR 模型推理 Demo")
    print("="*60)

    # 加载模型
    model, processor = load_model_and_processor(args.model_dir, args.base_model_dir)

    # 推理
    output = inference(
        model=model,
        processor=processor,
        image_path=args.image,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # 显示结果
    print("\n" + "="*60)
    print("📝 模型输出")
    print("="*60)

    if args.show_full:
        print("\n【完整输出】:")
        print(output)
        print("\n" + "-"*60)

    # 提取并显示答案
    answer = extract_answer(output)
    print("\n【提取的答案】:")
    print(answer)

    print("\n" + "="*60)
    print("✅ 完成")
    print("="*60)


if __name__ == "__main__":
    main()
