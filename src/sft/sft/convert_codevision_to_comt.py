#!/usr/bin/env python3
"""
Convert CodeVision-SFT format to COMT format
- First image is input image
- Remaining images are helper images
- Extract only GPT responses (no tool calls)
"""

import json
import argparse
import os
import re
from typing import List, Dict, Any


def extract_answer_from_gpt(gpt_response: str) -> str:
    """Extract final answer from <answer></answer> tags"""
    match = re.search(r'<answer>(.*?)</answer>', gpt_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return gpt_response.strip()


def extract_text_content(gpt_response: str) -> str:
    """Extract text content from GPT response, excluding tool calls"""
    # Remove <tool_call>...</tool_call> blocks
    text = re.sub(r'<tool_call>.*?</tool_call>', '', gpt_response, flags=re.DOTALL)

    # Extract <think> content
    think_matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)

    # Extract <answer> content
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

    # Combine think and answer content
    content_parts = []
    for think in think_matches:
        content_parts.append(think.strip())

    if answer_match:
        content_parts.append(answer_match.group(1).strip())

    return ' '.join(content_parts) if content_parts else text.strip()


def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single CodeVision sample to COMT format"""

    conversations = sample.get('conversations', [])
    images = sample.get('image_input', sample.get('images', []))

    # Extract question from first human message
    question = ""
    for conv in conversations:
        if conv.get('from') == 'human':
            # Remove <image> tags
            question = conv.get('value', '').replace('<image>', '').strip()
            break

    # Extract GPT responses (skip tool responses)
    gpt_responses = []
    for conv in conversations:
        if conv.get('from') == 'gpt':
            text = extract_text_content(conv.get('value', ''))
            if text:
                gpt_responses.append(text)

    # Extract final answer from last GPT response
    final_answer = ""
    if gpt_responses:
        last_response = conversations[-1].get('value', '') if conversations else ''
        final_answer = extract_answer_from_gpt(last_response)

    # Build sequence_plan
    # Pattern: gpt_response_1 -> helper_image_1 -> gpt_response_2 -> helper_image_2 -> ...
    sequence_plan = []

    helper_image_idx = 1  # Start from second image (first is input)

    for conv in conversations:
        if conv.get('from') == 'gpt':
            # Get GPT response and remove tool calls
            response = conv.get('value', '')
            text = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
            text = text.strip()

            if text:
                # Add GPT response
                sequence_plan.append({
                    "type": "text",
                    "content": text
                })

                # Add corresponding helper image if available
                if helper_image_idx < len(images):
                    sequence_plan.append({
                        "type": "latent",
                        "helper_image": images[helper_image_idx]
                    })
                    helper_image_idx += 1

    # Build COMT format
    comt_sample = {
        "text_input": question,
        "image_input": [images[0]] if images else [],  # First image only
        "sequence_plan": sequence_plan,
        "original_final_answer": final_answer
    }

    return comt_sample


def main():
    parser = argparse.ArgumentParser(
        description="Convert CodeVision-SFT format to COMT format"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input JSON file in CodeVision format"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output JSONL file in COMT format"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to convert (for testing)"
    )

    args = parser.parse_args()

    # Load CodeVision data
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        codevision_data = json.load(f)

    print(f"Loaded {len(codevision_data)} samples")

    # Apply limit if specified
    if args.limit:
        codevision_data = codevision_data[:args.limit]
        print(f"Processing first {args.limit} samples")

    # Convert samples
    comt_samples = []
    for i, sample in enumerate(codevision_data):
        try:
            comt_sample = convert_sample(sample)
            comt_samples.append(comt_sample)

            if (i + 1) % 100 == 0:
                print(f"Converted {i + 1}/{len(codevision_data)} samples")

        except Exception as e:
            print(f"Error converting sample {i}: {e}")
            continue

    # Save to JSONL
    print(f"\nSaving {len(comt_samples)} samples to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in comt_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Conversion complete!")
    print(f"Output saved to: {args.output_file}")

    # Print sample for verification
    if comt_samples:
        print("\n" + "="*60)
        print("Sample conversion (first entry):")
        print("="*60)
        print(json.dumps(comt_samples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
