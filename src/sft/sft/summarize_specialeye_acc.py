#!/usr/bin/env python3
"""
计算 SpecialEye Benchmark 准确度 - ILVR 模型

适配输出格式: <answer>[(0.114, 0.252)]</answer>
"""
import os
import re
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def extract_answer_from_tags(text: str) -> str:
    """
    从 <answer></answer> 标签中提取答案

    Args:
        text: 模型输出文本

    Returns:
        提取的答案内容
    """
    # 尝试匹配 <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 如果没有标签，返回整个文本
    return text.strip()


def parse_coordinates(answer_text: str, width=640, height=480, return_normalized=False):
    """
    从答案中解析坐标点

    支持多种格式:
    1. [(0.114, 0.252)]
    2. [(x1, y1), (x2, y2)]
    3. [{"point_2d": [x, y], "label": "point_1"}]
    4. {"point_2d": [[x1, y1], [x2, y2]]}  # 新格式
    5. (0.114, 0.252)

    Args:
        answer_text: 从<answer>标签提取的文本
        width: 图片宽度
        height: 图片高度
        return_normalized: 如果为True，返回归一化坐标(0-1)；否则返回像素坐标

    Returns:
        numpy array of shape (N, 2) 包含坐标（归一化或像素坐标）
    """
    try:
        answer_text = answer_text.strip()

        # 方式1: JSON格式 - 检测是否包含 point_2d
        if "{" in answer_text and "point_2d" in answer_text:
            # 清理markdown
            if "```json" in answer_text:
                answer_text = answer_text.split("```json")[1].split("```")[0]

            data = json.loads(answer_text)
            points = []

            # 新格式: {"point_2d": [[x1, y1], [x2, y2]]}
            if isinstance(data, dict) and "point_2d" in data:
                point_data = data["point_2d"]

                # point_2d 可能是 [[x, y]] 或 [x, y]
                if isinstance(point_data, list):
                    if len(point_data) > 0:
                        # 检查是单点 [x, y] 还是多点 [[x1, y1], [x2, y2]]
                        if isinstance(point_data[0], list):
                            # 多点格式: [[x1, y1], [x2, y2]]
                            for point in point_data:
                                x, y = point[0], point[1]
                                # 判断是归一化还是像素坐标
                                if x > 1.0 or y > 1.0:
                                    # 像素坐标，需要归一化
                                    x_norm = x / width
                                    y_norm = y / height
                                else:
                                    # 已经是归一化坐标
                                    x_norm = x
                                    y_norm = y

                                if return_normalized:
                                    points.append([x_norm, y_norm])
                                else:
                                    points.append([int(x_norm * width), int(y_norm * height)])
                        else:
                            # 单点格式: [x, y]
                            x, y = point_data[0], point_data[1]
                            if x > 1.0 or y > 1.0:
                                x_norm = x / width
                                y_norm = y / height
                            else:
                                x_norm = x
                                y_norm = y

                            if return_normalized:
                                points.append([x_norm, y_norm])
                            else:
                                points.append([int(x_norm * width), int(y_norm * height)])

                return np.array(points)

            # 旧格式: [{"point_2d": [x, y], "label": "point_1"}]
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "point_2d" in item:
                        x, y = item["point_2d"][0], item["point_2d"][1]
                        # 判断是归一化还是像素坐标
                        if x > 1.0 or y > 1.0:
                            # 像素坐标，需要归一化
                            x_norm = x / width
                            y_norm = y / height
                        else:
                            # 已经是归一化坐标
                            x_norm = x
                            y_norm = y

                        if return_normalized:
                            points.append([x_norm, y_norm])
                        else:
                            points.append([int(x_norm * width), int(y_norm * height)])

                return np.array(points)

        # 方式2: 元组列表格式 [(x1, y1), (x2, y2)]
        tuple_pattern = r'\(([0-9.]+),\s*([0-9.]+)\)'
        matches = re.findall(tuple_pattern, answer_text)

        if matches:
            points = []
            for x_str, y_str in matches:
                x = float(x_str)
                y = float(y_str)

                # 判断是归一化坐标还是像素坐标
                if x > 1.0 or y > 1.0:
                    # 像素坐标，归一化
                    x_norm = x / width
                    y_norm = y / height
                else:
                    # 已经是归一化坐标
                    x_norm = x
                    y_norm = y

                if return_normalized:
                    points.append([x_norm, y_norm])
                else:
                    points.append([int(x_norm * width), int(y_norm * height)])

            return np.array(points)

        # 方式3: 列表格式 [x, y] 或 [[x, y]]
        try:
            data = json.loads(answer_text)
            if isinstance(data, list):
                if len(data) == 2 and isinstance(data[0], (int, float)):
                    # 单点: [x, y]
                    x, y = data
                    if x > 1.0 or y > 1.0:
                        x_norm = x / width
                        y_norm = y / height
                    else:
                        x_norm = x
                        y_norm = y

                    if return_normalized:
                        return np.array([[x_norm, y_norm]])
                    else:
                        return np.array([[int(x_norm * width), int(y_norm * height)]])

                elif len(data) > 0 and isinstance(data[0], list):
                    # 多点: [[x1, y1], [x2, y2]]
                    points = []
                    for point in data:
                        x, y = point
                        if x > 1.0 or y > 1.0:
                            x_norm = x / width
                            y_norm = y / height
                        else:
                            x_norm = x
                            y_norm = y

                        if return_normalized:
                            points.append([x_norm, y_norm])
                        else:
                            points.append([int(x_norm * width), int(y_norm * height)])
                    return np.array(points)
        except:
            pass

        print(f"⚠️  无法解析坐标: {answer_text[:100]}")
        return np.array([])

    except Exception as e:
        print(f"❌ 解析错误: {e}, 文本: {answer_text[:100]}")
        return np.array([])


def bilinear_interpolate(mask, x, y):
    """
    双线性插值获取mask值

    Args:
        mask: 2D numpy array
        x, y: 浮点坐标

    Returns:
        插值后的mask值
    """
    h, w = mask.shape

    # 确保坐标在范围内
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    # 获取四个邻近点
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, h - 1)

    # 计算权重
    wx = x - x0
    wy = y - y0

    # 双线性插值
    val = (1 - wx) * (1 - wy) * mask[y0, x0] + \
          wx * (1 - wy) * mask[y0, x1] + \
          (1 - wx) * wy * mask[y1, x0] + \
          wx * wy * mask[y1, x1]

    return val


def visualize_result(rgb_path, mask_path, points_norm, accuracy, question_info, output_path):
    """
    可视化结果：显示RGB图像、Mask、预测点和准确度

    Args:
        rgb_path: RGB图像路径
        mask_path: Mask图像路径
        points_norm: 归一化坐标点 (0-1)
        accuracy: 准确度值
        question_info: 问题信息字典
        output_path: 输出路径
    """
    try:
        # 加载图像
        rgb_img = Image.open(rgb_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        w, h = rgb_img.size
        mask_array = np.array(mask_img) / 255.0
        mask_binary = (mask_array > 0).astype(np.uint8)

        # 转换归一化坐标到像素坐标
        points_pixel = []
        point_in_mask = []
        for x_norm, y_norm in points_norm:
            x_px = x_norm * w
            y_px = y_norm * h
            points_pixel.append([x_px, y_px])

            # 判断点是否在mask内
            x_int = int(np.clip(x_px, 0, w - 1))
            y_int = int(np.clip(y_px, 0, h - 1))
            in_mask = mask_binary[y_int, x_int] > 0
            point_in_mask.append(in_mask)

        # 创建图形
        fig = plt.figure(figsize=(20, 8))

        # 1. RGB图像
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(rgb_img)
        ax1.set_title("RGB Image", fontsize=14, fontweight='bold')
        ax1.set_xlabel(f"Size: {w} × {h}", fontsize=10)
        ax1.axis('off')

        # 2. Mask图像
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(mask_img, cmap='gray')
        ax2.set_title("Ground Truth Mask", fontsize=14, fontweight='bold')
        ax2.axis('off')

        # 3. RGB + 预测点
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(rgb_img)

        # 绘制预测点
        for i, (point, in_mask) in enumerate(zip(points_pixel, point_in_mask)):
            x_px, y_px = point
            color = 'green' if in_mask else 'red'
            status = "✓" if in_mask else "✗"

            # 绘制点
            circle = Circle((x_px, y_px), radius=10, color=color, fill=True, alpha=0.7)
            ax3.add_patch(circle)

            # 添加标签
            label_text = f'P{i+1} {status}'
            ax3.text(x_px, y_px-20, label_text, color=color, fontsize=10,
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color))

        ax3.set_title(f"Predictions (Acc: {accuracy:.2%})", fontsize=14, fontweight='bold')
        ax3.axis('off')

        # 添加总标题
        title = f"Question {question_info.get('question_id', 'N/A')} - "
        title += f"Category: {question_info.get('category', 'N/A')} - "
        title += f"Accuracy: {accuracy:.4f}"
        plt.suptitle(title, fontsize=16, fontweight='bold')

        # 添加图例
        fig.text(0.02, 0.02,
                f"Legend: ✓ = Point in mask (Green), ✗ = Point outside mask (Red)\n"
                f"Prompt: {question_info.get('prompt', 'N/A')[:80]}...",
                fontsize=9, style='italic')

        # 保存
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"⚠️  可视化失败: {e}")
        import traceback
        traceback.print_exc()


def compute_accuracy(
    answers: List[Dict],
    task_name: str,
    benchmark_dir: str = "/ytech_m2v_hdd/wangyuji/data/JingkunAn/RefSpatial-Bench",
    use_normalized=True,
    use_interpolation=True,
    visualize=False,
    output_dir=None
) -> Dict:
    """
    计算准确度

    Args:
        answers: 答案列表
        task_name: 任务名称
        benchmark_dir: benchmark 数据目录
        use_normalized: 是否使用归一化坐标计算（推荐True）
        use_interpolation: 是否使用双线性插值（推荐True）
        visualize: 是否生成可视化结果
        output_dir: 可视化结果输出目录

    Returns:
        准确度统计信息
    """
    accuracy_list = []
    failed_count = 0
    category_acc = {}

    # 创建可视化目录
    if visualize and output_dir:
        vis_dir = os.path.join(output_dir, f"{task_name}_visualizations")
        os.makedirs(vis_dir, exist_ok=True)

    for answer in tqdm(answers, desc="Computing accuracy"):
        # 加载 mask
        mask_path = os.path.join(benchmark_dir, task_name, answer['mask_path'])

        if not os.path.exists(mask_path):
            print(f"⚠️  Mask 不存在: {mask_path}")
            failed_count += 1
            continue

        mask = np.array(Image.open(mask_path)) / 255.0
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.float32)  # 保持为float以支持插值

        h, w = mask.shape

        # 提取答案
        answer_content = extract_answer_from_tags(answer["text"])

        # 解析坐标
        points_norm = np.array([])
        points_pixel = np.array([])
        try:
            if use_normalized:
                # 获取归一化坐标 (0-1)
                points_norm = parse_coordinates(answer_content, w, h, return_normalized=True)
            else:
                # 获取像素坐标
                points_pixel = parse_coordinates(answer_content, w, h, return_normalized=False)
        except Exception as e:
            print(f"⚠️  解析失败 question {answer['question_id']}: {e}")
            failed_count += 1
            answer["accuracy"] = 0.0
            answer["parsed_points"] = []
            accuracy_list.append(0.0)
            continue

        # 计算准确度
        acc = 0.0

        if use_normalized and len(points_norm) > 0:
            # 方法1：归一化坐标 + 双线性插值（最精确）
            points = points_norm

            # 检查坐标是否在有效范围 [0, 1]
            valid_mask = (points[:, 0] >= 0) & (points[:, 0] <= 1) & \
                        (points[:, 1] >= 0) & (points[:, 1] <= 1)

            if valid_mask.sum() > 0:
                valid_points = points[valid_mask]

                if use_interpolation:
                    # 使用双线性插值
                    mask_values = []
                    for x_norm, y_norm in valid_points:
                        # 转换为像素坐标（浮点）
                        x_pixel = x_norm * w
                        y_pixel = y_norm * h
                        # 插值
                        val = bilinear_interpolate(mask, x_pixel, y_pixel)
                        mask_values.append(val)
                    acc = np.mean(mask_values)
                else:
                    # 最近邻采样
                    x_pixel = (valid_points[:, 0] * w).astype(int)
                    y_pixel = (valid_points[:, 1] * h).astype(int)
                    x_pixel = np.clip(x_pixel, 0, w - 1)
                    y_pixel = np.clip(y_pixel, 0, h - 1)
                    mask_values = mask[y_pixel, x_pixel]
                    acc = mask_values.mean()

                # 考虑无效点
                total_acc = (acc * valid_mask.sum() + 0.0 * (~valid_mask).sum()) / len(points)
                acc = total_acc

            answer["parsed_points"] = points.tolist()

        elif not use_normalized and len(points_pixel) > 0:
            # 方法2：像素坐标
            points = points_pixel

            # 检查点是否在图像范围内
            in_range = (points[:, 0] >= 0) & (points[:, 0] < w) & \
                       (points[:, 1] >= 0) & (points[:, 1] < h)

            if in_range.sum() > 0:
                valid_points = points[in_range]
                mask_values = mask[valid_points[:, 1], valid_points[:, 0]]
                acc = mask_values.mean()

                # 考虑范围外的点
                total_acc = (acc * in_range.sum() + 0.0 * (~in_range).sum()) / len(points)
                acc = total_acc

            answer["parsed_points"] = points.tolist()

        answer["accuracy"] = float(acc)
        accuracy_list.append(acc)

        # 生成可视化
        if visualize and output_dir and use_normalized and len(points_norm) > 0:
            rgb_path = os.path.join(benchmark_dir, task_name, answer.get('rgb_path', ''))
            if os.path.exists(rgb_path):
                vis_output_path = os.path.join(vis_dir, f"question_{answer['question_id']}.png")
                question_info = {
                    'question_id': answer['question_id'],
                    'category': answer.get('category', 'unknown'),
                    'prompt': answer.get('prompt', 'N/A')
                }
                visualize_result(rgb_path, mask_path, points_norm, acc, question_info, vis_output_path)

        # 按类别统计
        category = answer.get("category", "unknown")
        if category not in category_acc:
            category_acc[category] = []
        category_acc[category].append(acc)

    # 计算统计信息
    stats = {
        "overall_accuracy": float(np.mean(accuracy_list)) if accuracy_list else 0.0,
        "evaluated": len(accuracy_list),
        "total": len(answers),
        "failed": failed_count,
        "category_accuracy": {cat: float(np.mean(accs)) for cat, accs in category_acc.items()},
        "config": {
            "use_normalized": use_normalized,
            "use_interpolation": use_interpolation
        }
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="计算 SpecialEye Benchmark 准确度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算单个任务的准确度
  python summarize_specialeye_acc.py \\
    --model_name step_2000 \\
    --task_name Location \\
    --output_folder ./benchmark_outputs

  # 计算所有任务的准确度并生成可视化
  python summarize_specialeye_acc.py \\
    --model_name step_2000 \\
    --task_name Location Placement Unseen \\
    --output_folder ./benchmark_outputs \\
    --visualize
        """
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="模型名称（输出目录下的子目录名）"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        nargs="+",
        required=True,
        help="任务名称，可选: Location, Placement, Unseen"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./benchmark_outputs",
        help="结果输出目录"
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="/ytech_m2v_hdd/wangyuji/data/JingkunAn/RefSpatial-Bench",
        help="Benchmark 数据目录"
    )
    parser.add_argument(
        "--use_pixel",
        action="store_true",
        help="使用像素坐标而非归一化坐标（默认使用归一化坐标）"
    )
    parser.add_argument(
        "--no_interpolation",
        action="store_true",
        help="不使用双线性插值（默认使用插值获得更精确结果）"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="生成可视化结果（默认不生成）"
    )

    args = parser.parse_args()

    # 确定使用的坐标系统
    use_normalized = not args.use_pixel
    use_interpolation = not args.no_interpolation

    print("="*60)
    print("📊 SpecialEye Benchmark 准确度计算")
    print("="*60)
    print(f"坐标系统: {'归一化 (0-1)' if use_normalized else '像素坐标'}")
    print(f"插值方法: {'双线性插值' if use_interpolation else '最近邻'}")
    print(f"可视化: {'启用' if args.visualize else '禁用'}")
    print("="*60)

    all_stats = {}

    for task_name in args.task_name:
        print(f"\n{'='*60}")
        print(f"📋 任务: {task_name}")
        print(f"{'='*60}")

        # 读取结果文件
        answer_file = os.path.join(args.output_folder, args.model_name, f"{task_name}.jsonl")

        if not os.path.exists(answer_file):
            print(f"❌ 结果文件不存在: {answer_file}")
            continue

        with open(answer_file, 'r', encoding='utf-8') as f:
            answers = [json.loads(line) for line in f]

        print(f"📄 加载了 {len(answers)} 个结果")

        # 计算准确度
        stats = compute_accuracy(
            answers,
            task_name,
            args.benchmark_dir,
            use_normalized=use_normalized,
            use_interpolation=use_interpolation,
            visualize=args.visualize,
            output_dir=os.path.join(args.output_folder, args.model_name)
        )
        all_stats[task_name] = stats

        # 打印结果
        print(f"\n{'='*60}")
        print(f"📊 {task_name} 结果:")
        print(f"{'='*60}")
        print(f"配置: 坐标系统={stats['config']['use_normalized']}, 插值={stats['config']['use_interpolation']}")
        print(f"总体准确度: {stats['overall_accuracy']:.4f}")
        print(f"评估样本数: {stats['evaluated']}")
        print(f"总样本数: {stats['total']}")
        print(f"失败样本数: {stats['failed']}")

        if stats['category_accuracy']:
            print(f"\n按类别准确度:")
            for cat, acc in sorted(stats['category_accuracy'].items()):
                print(f"  {cat}: {acc:.4f}")

        # 保存带准确度的结果
        output_with_acc = answer_file.replace('.jsonl', '_with_acc.jsonl')
        with open(output_with_acc, 'w', encoding='utf-8') as f:
            for answer in answers:
                f.write(json.dumps(answer, ensure_ascii=False) + "\n")
        print(f"\n💾 已保存带准确度的结果: {output_with_acc}")

        # 如果启用了可视化，打印可视化目录
        if args.visualize:
            vis_dir = os.path.join(args.output_folder, args.model_name, f"{task_name}_visualizations")
            print(f"🖼️  可视化结果已保存至: {vis_dir}")

    # 打印总结
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print("📊 总体统计")
        print(f"{'='*60}")
        for task, stats in all_stats.items():
            print(f"{task:15s}: {stats['overall_accuracy']:.4f} ({stats['evaluated']}/{stats['total']})")

        # 平均准确度
        avg_acc = np.mean([stats['overall_accuracy'] for stats in all_stats.values()])
        print(f"\n{'='*60}")
        print(f"平均准确度: {avg_acc:.4f}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
