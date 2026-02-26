#!/usr/bin/env python3
"""
ILVR Benchmark Evaluation Script
Evaluates ILVR model on BLINK, VSTAR, and MMVP benchmarks
"""

import os
import json
import argparse
import re
import csv
import string
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset


def load_ilvr_model(model_path: str, device: str = "cuda"):
    """Load ILVR model and processor"""
    print(f"Loading ILVR model from {model_path}...")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print("Processor loaded successfully!")

    # Load model
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        print("Model loaded with flash_attention_2")
    except Exception as e:
        print(f"Flash Attention 2 failed, using eager mode: {e}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        print("Model loaded with eager attention")

    model.eval()
    print("Model loaded successfully!")
    return model, processor


def query_ilvr_model(
    model,
    processor,
    messages: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Query ILVR model with messages"""

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def load_vstar_dataset(benchmark_dir: str) -> List[Dict]:
    """Load VSTAR benchmark dataset"""
    vstar_path = os.path.join(benchmark_dir, "vstar_bench")
    data_file = os.path.join(vstar_path, "test_questions.jsonl")

    samples = []
    with open(data_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())

            # Image path is relative in JSONL
            image_path = os.path.join(vstar_path, item['image'])

            # Text already contains question and choices
            full_question = item['text'] + "\n\nPut your answer in <answer></answer> tags."

            samples.append({
                'id': item['question_id'],
                'image_path': image_path,
                'question': full_question,
                'ground_truth': item['label'].upper(),
                'category': item.get('category', 'unknown')
            })

    return samples


def load_mmvp_dataset(benchmark_dir: str) -> List[Dict]:
    """Load MMVP benchmark dataset"""
    mmvp_path = os.path.join(benchmark_dir, "MMVP")
    csv_file = os.path.join(mmvp_path, "Questions.csv")
    image_dir = os.path.join(mmvp_path, "MMVP Images")

    samples = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["Index"])
            image_name = f"{idx}.jpg"
            image_path = os.path.join(image_dir, image_name)

            # Build question with options
            full_question = f"{row['Question']}\nOptions:\n{row['Options']}\n\nAnswer with the option's letter from the given choices directly. Put your answer in <answer></answer> tags."

            samples.append({
                'id': str(idx),
                'image_path': image_path,
                'question': full_question,
                'ground_truth': row["Correct Answer"].strip().upper(),
                'category': 'mmvp'
            })

    return samples


def load_blink_dataset(benchmark_dir: str, split: str = "val") -> List[Dict]:
    """Load BLINK benchmark dataset"""
    blink_path = os.path.join(benchmark_dir, "BLINK")

    # Load specific BLINK configs from local HuggingFace dataset directory
    configs = ['Counting', 'IQ_Test', 'Jigsaw', 'Relative_Reflectance', 'Spatial_Relation']
    all_datasets = {}

    for config in configs:
        # Load from local HuggingFace dataset directory with specific config
        all_datasets[config] = load_dataset(blink_path, config, trust_remote_code=True)

    samples = []
    for config in all_datasets:
        ds = all_datasets[config][split]
        for dat in ds:
            idx = dat["idx"]
            choices = dat["choices"]
            letters = string.ascii_uppercase
            paired = list(zip(letters, choices))
            option_string = ""
            for letter, choice in paired:
                option_string += f"{letter}. {choice}\n"

            # Extract answer (handle multi-character answers)
            if len(dat['answer']) > 1:
                ans = dat['answer'][1].upper()
            else:
                ans = dat['answer'][0].upper()

            # Collect images (BLINK can have multiple images)
            images = []
            for k in ['image_1', 'image_2', 'image_3', 'image_4']:
                if k in dat and dat[k] is not None:
                    images.append(dat[k])

            # Build question
            question = dat['question'] + "\nOptions:\n" + option_string
            question += "\nAnswer with the option's letter from the given choices directly. Put your answer in <answer></answer> tags."

            samples.append({
                'id': idx,
                'images': images,  # Note: multiple images
                'question': question,
                'ground_truth': ans,
                'category': config
            })

    return samples


def extract_answer(response: str) -> str:
    """Extract answer from <answer></answer> tags"""
    answer_match = re.search(r'<answer>\s*([A-Z])\s*</answer>', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Fallback: try to find single letter answer
    single_letter = re.search(r'\b([A-Z])\b', response)
    if single_letter:
        return single_letter.group(1)

    return ""


def evaluate_benchmark(
    model,
    processor,
    samples: List[Dict],
    benchmark_name: str,
    output_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> Dict:
    """Evaluate model on a benchmark"""

    print(f"\n{'='*60}")
    print(f"Evaluating {benchmark_name} ({len(samples)} samples)")
    print(f"{'='*60}\n")

    results = []
    correct = 0
    total = 0

    category_stats = {}

    for sample in tqdm(samples, desc=f"Processing {benchmark_name}"):
        # Prepare messages
        # Handle both single image (VSTAR, MMVP) and multiple images (BLINK)
        content = []

        if 'image_path' in sample:
            # Single image
            content.append({"type": "image", "image": f"file://{sample['image_path']}"})
        elif 'images' in sample:
            # Multiple images (BLINK)
            for img in sample['images']:
                # Images in BLINK are PIL objects from datasets
                content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": sample['question']})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        # Query model
        try:
            response = query_ilvr_model(
                model,
                processor,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            predicted_answer = extract_answer(response)
            ground_truth = sample['ground_truth'].strip().upper()

            is_correct = (predicted_answer == ground_truth)

            if is_correct:
                correct += 1
            total += 1

            # Category stats
            category = sample.get('category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {'correct': 0, 'total': 0}
            category_stats[category]['total'] += 1
            if is_correct:
                category_stats[category]['correct'] += 1

            results.append({
                'id': sample['id'],
                'question': sample['question'],
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'full_response': response,
                'correct': is_correct,
                'category': category,
            })

        except Exception as e:
            print(f"\nError processing sample {sample['id']}: {e}")
            results.append({
                'id': sample['id'],
                'question': sample['question'],
                'ground_truth': sample['ground_truth'],
                'predicted_answer': "",
                'full_response': f"ERROR: {str(e)}",
                'correct': False,
                'category': sample.get('category', 'unknown'),
            })

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Calculate category accuracies
    category_accuracies = {}
    for cat, stats in category_stats.items():
        category_accuracies[cat] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"{benchmark_name}_results_{timestamp}.json")

    summary = {
        'benchmark': benchmark_name,
        'timestamp': timestamp,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'category_stats': category_stats,
        'category_accuracies': category_accuracies,
        'results': results,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{benchmark_name} Results:")
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"\nCategory Breakdown:")
    for cat, acc in category_accuracies.items():
        stats = category_stats[cat]
        print(f"  {cat}: {stats['correct']}/{stats['total']} = {acc:.2%}")
    print(f"\nResults saved to: {result_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate ILVR model on benchmarks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to ILVR model")
    parser.add_argument("--benchmark_dir", type=str, default="/ytech_m2v_hdd/wangyuji/data/lvt/bench",
                        help="Base directory for benchmarks")
    parser.add_argument("--output_dir", type=str, default="/ytech_m2v_hdd/wangyuji/code/ILVR/results",
                        help="Output directory for results")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["vstar", "mmvp", "blink"],
                        choices=["vstar", "mmvp", "blink"], help="Benchmarks to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, processor = load_ilvr_model(args.model_path, args.device)

    # Evaluate each benchmark
    all_summaries = []

    for benchmark_name in args.benchmarks:
        print(f"\nLoading {benchmark_name} dataset...")

        if benchmark_name == "vstar":
            samples = load_vstar_dataset(args.benchmark_dir)
        elif benchmark_name == "mmvp":
            samples = load_mmvp_dataset(args.benchmark_dir)
        elif benchmark_name == "blink":
            samples = load_blink_dataset(args.benchmark_dir, split="val")
        else:
            print(f"Unknown benchmark: {benchmark_name}")
            continue

        summary = evaluate_benchmark(
            model,
            processor,
            samples,
            benchmark_name,
            args.output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        all_summaries.append(summary)

    # Print overall summary
    print(f"\n{'='*60}")
    print("Overall Summary")
    print(f"{'='*60}")
    for summary in all_summaries:
        print(f"{summary['benchmark']}: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total_samples']})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
