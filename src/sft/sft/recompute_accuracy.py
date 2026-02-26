#!/usr/bin/env python3
"""
Recompute accuracy for benchmark results with corrected answer matching
Fixes issue where ground truth has parentheses like "(B)" but predicted is "B"
"""

import json
import argparse
import os


def normalize_answer(answer: str) -> str:
    """Normalize answer by removing parentheses and whitespace, converting to uppercase"""
    answer = answer.strip().upper()
    # Remove parentheses
    answer = answer.replace('(', '').replace(')', '')
    # Take first character if multi-char
    if len(answer) > 0:
        answer = answer[0]
    return answer


def recompute_accuracy(result_file: str, output_file: str = None):
    """Recompute accuracy with normalized answer matching"""

    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])
    total = len(results)
    correct = 0

    # Category statistics
    category_stats = {}

    # Track changes
    changed_count = 0

    for result in results:
        ground_truth = normalize_answer(result['ground_truth'])
        predicted_answer = normalize_answer(result['predicted_answer'])

        is_correct = (predicted_answer == ground_truth)

        # Check if correction status changed
        if result.get('correct') != is_correct:
            changed_count += 1
            print(f"ID {result['id']}: GT='{result['ground_truth']}' -> '{ground_truth}', "
                  f"Pred='{result['predicted_answer']}' -> '{predicted_answer}', "
                  f"Changed from {result.get('correct')} to {is_correct}")

        # Update result
        result['correct'] = is_correct

        if is_correct:
            correct += 1

        # Category stats
        category = result.get('category', 'unknown')
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0}
        category_stats[category]['total'] += 1
        if is_correct:
            category_stats[category]['correct'] += 1

    # Calculate accuracies
    accuracy = correct / total if total > 0 else 0.0
    category_accuracies = {}
    for cat, stats in category_stats.items():
        category_accuracies[cat] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

    # Update summary
    data['total_samples'] = total
    data['correct'] = correct
    data['accuracy'] = accuracy
    data['category_stats'] = category_stats
    data['category_accuracies'] = category_accuracies

    # Print summary
    print(f"\n{'='*60}")
    print(f"Recomputed Results for: {os.path.basename(result_file)}")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Changed results: {changed_count}")

    if category_stats:
        print(f"\nCategory Breakdown:")
        for cat, stats in category_stats.items():
            acc = category_accuracies[cat]
            print(f"  {cat}: {stats['correct']}/{stats['total']} = {acc:.2%}")

    # Save updated results
    if output_file is None:
        # Generate output filename
        base_name = os.path.basename(result_file)
        dir_name = os.path.dirname(result_file)
        output_file = os.path.join(dir_name, base_name.replace('.json', '_recomputed.json'))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated results saved to: {output_file}")
    print(f"{'='*60}\n")

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'category_stats': category_stats,
        'changed_count': changed_count
    }


def main():
    parser = argparse.ArgumentParser(description="Recompute accuracy for benchmark results")
    parser.add_argument("result_file", type=str, help="Path to result JSON file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: add _recomputed suffix)")

    args = parser.parse_args()

    if not os.path.exists(args.result_file):
        print(f"Error: File not found: {args.result_file}")
        return

    recompute_accuracy(args.result_file, args.output)


if __name__ == "__main__":
    main()
