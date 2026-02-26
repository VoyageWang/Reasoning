import re
import json
from scipy.optimize import linear_sum_assignment
import numpy as np

def vision_reasoner_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 1.0 if match else 0.0 
    
    def segmentation_format(predict_str: str) -> float:
        segmentation_format_reward = 0.0
        try:
            json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
            if not json_match:
                return segmentation_format_reward
            data = json.loads(json_match.group(1))
            
            data_cnt = len(data)
            
            for item in data:
                cur_reward = 0.0

                if 'bbox_2d' in item:
                    bbox_2d = item['bbox_2d']
                    if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        cur_reward += 1.0
                    
                if 'point_2d' in item:
                    point_2d = item['point_2d']
                    if isinstance(point_2d, list) and len(point_2d) == 2:
                        cur_reward += 1.0
                
                segmentation_format_reward += cur_reward / data_cnt
        except Exception:
            pass
        return segmentation_format_reward
        
    segmentation_format_reward = segmentation_format(predict_str)
    
    return thinking_format_reward + segmentation_format_reward

def vision_reasoner_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    max_accuracy_reward = 0.0
    MAX_OBJECTS = 120  # 设置上限
    
    try:
        gt_data = json.loads(ground_truth)
        gt_bboxes = [item['bbox_2d'] for item in gt_data]
        gt_points = [item['point_2d'] for item in gt_data]
            
        #json_match = re.search(r'```json\s*(.*?)\s*```', predict_str, re.DOTALL)
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            pred_bboxes = [item['bbox_2d'] for item in data]
            pred_points = [item['point_2d'] for item in data]
            
            # 只有当预测或真实值超过上限时才截断
            if len(pred_bboxes) > MAX_OBJECTS:
                pred_bboxes = pred_bboxes[:MAX_OBJECTS]
                pred_points = pred_points[:MAX_OBJECTS]
            
            if len(gt_bboxes) > MAX_OBJECTS:
                gt_bboxes = gt_bboxes[:MAX_OBJECTS]
                gt_points = gt_points[:MAX_OBJECTS]
            
            # 预处理数据为numpy数组
            pred_bboxes = np.array(pred_bboxes)  # (M,4)
            pred_points = np.array(pred_points)  # (M,2)
            gt_bboxes = np.array(gt_bboxes)    # (N,4)
            gt_points = np.array(gt_points)     # (N,2)
            
            # 并行计算所有指标
            iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # (M,N)
            l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)  # (M,N)
            points_dist_matrix = batch_points_distance(pred_points, gt_points)  # (M,N)
            points_in_box = batch_points_in_box(pred_points, pred_bboxes)  # (M,)
            
            # 计算reward矩阵
            iou_reward = (iou_matrix > 0.5).astype(float)
            bbox_l1_reward = (l1_matrix < 10).astype(float)
            point_reward = ((points_dist_matrix < 30) & points_in_box[:,np.newaxis]).astype(float)
            
            # 构建最终的cost矩阵
            cost_matrix = 3.0 - (iou_reward + bbox_l1_reward + point_reward)
            
            # 使用匈牙利算法找最优匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # 直接从cost_matrix计算总reward
            total_reward = len(row_indices) * 3.0 - cost_matrix[row_indices, col_indices].sum()
            
            # 计算平均reward
            max_length = max(len(pred_bboxes), len(gt_bboxes))
            max_accuracy_reward = total_reward / max_length
            
    except Exception:
        pass
    return max_accuracy_reward

def vision_reasoner_non_repeat_reward(predict_str: str) -> float:
    non_repeat_reward = 1.0  # 初始满分
    try:
        sentences = predict_str.split('.')

        # 移除空句子
        sentences = [s.strip() for s in sentences if s.strip()]

        # 检查重复
        seen = set()
        repeats = 0

        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >=2:
                non_repeat_reward = 0
                break
            seen.add(sentence)

    except Exception:
        pass

    return non_repeat_reward


def vision_reasoner_think_content_penalty(predict_str: str) -> float:
    """
    检查<think>标签中的内容是否只包含特殊token
    如果只有特殊token而没有实际文本内容，则扣除0.5分

    Args:
        predict_str: Model prediction string

    Returns:
        penalty: 0.0 (无惩罚) 或 -0.5 (有惩罚)
    """
    penalty = 0.0

    try:
        # 提取<think>标签中的内容
        think_match = re.search(r'<think>(.*?)</think>', predict_str, re.DOTALL)
        if not think_match:
            return penalty

        think_content = think_match.group(1).strip()

        # 定义所有特殊token
        special_tokens = [
            '<|sam_pad|>', '<|depth_pad|>', '<|dino_pad|>', '<|pidinet_pad|>',
            '<|sd_pad|>', '<|intern_pad|>', '<|siglip_pad|>', '<|metaclip_pad|>'
        ]

        # 移除所有特殊token
        content_without_tokens = think_content
        for token in special_tokens:
            content_without_tokens = content_without_tokens.replace(token, '')

        # 移除空白字符
        content_without_tokens = content_without_tokens.strip()

        # 如果移除特殊token后只剩下空白或标点符号，则施加惩罚
        # 保留一些常见的连接词和短语
        if len(content_without_tokens) == 0:
            penalty = -0.5
        elif all(c in ' \n\t.,;:!?、。，；：！？' for c in content_without_tokens):
            penalty = -0.5

    except Exception:
        pass

    return penalty


def vision_reasoner_token_context_penalty(predict_str: str) -> float:
    """
    检查特殊token是否出现在正确的上下文中：
    - <|sam_pad|> 应该只跟 'segmentation' 相关的词
    - <|depth_pad|> 应该只跟 'depth' 相关的词
    - <|dino_pad|> 应该只跟 'perception feature' 或 'feature' 相关的词
    - <|pidinet_pad|> 应该只跟 'edge' 相关的词

    检查特殊token之前的文本，确保前面出现的是对应的正确关键词
    如果发现token在错误的上下文中，扣除0.5分

    Args:
        predict_str: Model prediction string

    Returns:
        penalty: 0.0 或 -0.5
    """
    penalty = 0.0

    try:
        # 提取<think>标签中的内容
        think_match = re.search(r'<think>(.*?)</think>', predict_str, re.DOTALL)
        if not think_match:
            return penalty

        think_content = think_match.group(1).lower()  # 转小写以便匹配

        # 定义token和对应的正确上下文关键词
        token_context_rules = {
            '<|sam_pad|>': ['segmentation', 'segment', 'mask'],
            '<|depth_pad|>': ['depth', 'depth map'],
            '<|dino_pad|>': ['perception', 'perception feature', 'feature'],
            '<|pidinet_pad|>': ['edge', 'edge map']
        }

        has_violation = False

        for token, valid_keywords in token_context_rules.items():
            token_lower = token.lower()

            # 检查token是否存在
            if token_lower not in think_content:
                continue

            # 查找token出现的第一个位置
            token_pos = think_content.find(token_lower)
            if token_pos == -1:
                continue

            # 获取token之前的文本 (最多前100个字符，确保能捕获到描述词)
            context_start = max(0, token_pos - 20)
            preceding_text = think_content[context_start:token_pos]

            # 检查前面的文本中是否包含任何有效关键词
            has_valid_keyword = any(keyword in preceding_text for keyword in valid_keywords)

            if not has_valid_keyword:
                has_violation = True
                break

        if has_violation:
            penalty = -1

    except Exception:
        pass

    return penalty


def vision_reasoner_token_count_penalty(predict_str: str) -> float:
    """
    检查特殊token的出现次数是否符合要求：
    - <|sam_pad|> 应该恰好出现8次
    - <|depth_pad|> 应该恰好出现4次
    - <|dino_pad|> 应该恰好出现4次
    - <|pidinet_pad|> 应该恰好出现4次

    如果任何token的出现次数不符合要求，扣除1分

    Args:
        predict_str: Model prediction string

    Returns:
        penalty: 0.0 或 -1.0
    """
    penalty = 0.0

    try:
        # 提取<think>标签中的内容
        think_match = re.search(r'<think>(.*?)</think>', predict_str, re.DOTALL)
        if not think_match:
            return penalty

        think_content = think_match.group(1)

        # 定义每个token的期望出现次数
        expected_counts = {
            '<|sam_pad|>': 8,
            '<|depth_pad|>': 4,
            '<|dino_pad|>': 4,
            '<|pidinet_pad|>': 4
        }

        has_count_violation = False

        for token, expected_count in expected_counts.items():
            # 计算token的实际出现次数
            actual_count = think_content.count(token)

            # 如果次数不匹配，施加惩罚
            if actual_count != expected_count:
                has_count_violation = True
                break

        if has_count_violation:
            penalty = -1.0

    except Exception:
        pass

    return penalty


def vision_reasoner_token_phrase_matching_penalty(predict_str: str) -> float:
    """
    检查特定短语后面是否紧跟着正确的特殊token：
    - "the segmentation of the image" 后面应该恰好跟着 8个 <|sam_pad|>
    - "the depth map of the image" 后面应该恰好跟着 4个 <|depth_pad|>
    - "the perception feature of the image" 后面应该恰好跟着 4个 <|dino_pad|>

    如果发现短语后面跟的token不正确，扣除1分

    Args:
        predict_str: Model prediction string

    Returns:
        penalty: 0.0 或 -1.0
    """
    penalty = 0.0

    try:
        # 提取<think>标签中的内容
        think_match = re.search(r'<think>(.*?)</think>', predict_str, re.DOTALL)
        if not think_match:
            return penalty

        think_content = think_match.group(1)

        # 定义短语和对应的期望token模式
        phrase_token_rules = [
            {
                'phrase': r'the segmentation of the image\s+is\s+',
                'expected_token': '<|sam_pad|>',
                'expected_count': 8
            },
            {
                'phrase': r'the depth map of the image\s+is\s+',
                'expected_token': '<|depth_pad|>',
                'expected_count': 4
            },
            {
                'phrase': r'the perception feature of the image\s+is\s+',
                'expected_token': '<|dino_pad|>',
                'expected_count': 4
            },
            {
                'phrase': r'the edge map of the image\s+is\s+',
                'expected_token': '<|pidinet_pad|>',
                'expected_count': 4
            }

        ]

        has_violation = False

        for rule in phrase_token_rules:
            # 查找短语
            phrase_match = re.search(rule['phrase'], think_content, re.IGNORECASE)
            if not phrase_match:
                continue

            # 获取短语后面的内容
            start_pos = phrase_match.end()
            remaining_text = think_content[start_pos:]

            # 期望的token序列
            expected_sequence = rule['expected_token'] * rule['expected_count']

            # 检查是否以期望的token序列开头
            if not remaining_text.startswith(expected_sequence):
                has_violation = True
                break

            # 检查期望序列后面是否紧跟着其他token（不应该有多余的相同token）
            after_sequence = remaining_text[len(expected_sequence):]
            if after_sequence.startswith(rule['expected_token']):
                # 如果后面还有相同的token，说明数量不对
                has_violation = True
                break

        if has_violation:
            penalty = -1.0

    except Exception:
        pass

    return penalty

def vision_reasoner_compute_score(predict_str: str, ground_truth: str, verbose: bool = False) -> float:
    """
    Compute reward score for vision reasoner predictions.

    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth annotations
        verbose: If True, print detailed reward breakdown
    """
    format_reward = vision_reasoner_format_reward(predict_str)
    accuracy_reward = vision_reasoner_accuracy_reward(predict_str, ground_truth)
    non_repeat_reward = vision_reasoner_non_repeat_reward(predict_str)

    # 新增的penalties
    think_content_penalty = vision_reasoner_think_content_penalty(predict_str)
    token_context_penalty = vision_reasoner_token_context_penalty(predict_str)
    token_count_penalty = vision_reasoner_token_count_penalty(predict_str)
    token_phrase_matching_penalty = vision_reasoner_token_phrase_matching_penalty(predict_str)

    # reward = format_reward + accuracy_reward + non_repeat_reward + think_content_penalty + token_context_penalty + token_count_penalty
    
    reward = format_reward + accuracy_reward  + non_repeat_reward + think_content_penalty + token_phrase_matching_penalty

    # Print detailed breakdown periodically
    if verbose and hasattr(vision_reasoner_compute_score, '_call_count'):
        vision_reasoner_compute_score._call_count += 1
    elif verbose:
        vision_reasoner_compute_score._call_count = 1

    if verbose and vision_reasoner_compute_score._call_count % 50 == 0:
        print("\n" + "="*60)
        print(f"🎯 REWARD BREAKDOWN (Call #{vision_reasoner_compute_score._call_count})")
        print("="*60)
        print(f"Format Reward:           {format_reward:.4f}")
        print(f"Accuracy Reward:         {accuracy_reward:.4f}")
        print(f"Non-repeat Reward:       {non_repeat_reward:.4f}")
        print(f"Think Content Penalty:   {think_content_penalty:.4f}")
        print(f"Token Context Penalty:   {token_context_penalty:.4f}")
        print(f"Token Count Penalty:     {token_count_penalty:.4f}")
        print(f"Total Reward:            {reward:.4f}")
        print(f"\nPrediction preview: {predict_str[:200]}...")
        print("="*60 + "\n")

    return reward

def batch_iou(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    # 广播机制自动扩展维度
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # (M,1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # (N,1)
    
    xA = np.maximum(x11, np.transpose(x21))  # (M,N)
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)  # (M,1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)  # (N,1)
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / unionArea  # (M,N)
    return iou

def batch_l1_distance(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    boxes1 = boxes1[:, np.newaxis, :]  # (M,1,4)
    boxes2 = boxes2[np.newaxis, :, :]  # (1,N,4)
    return np.mean(np.abs(boxes1 - boxes2), axis=2)  # (M,N)

def batch_points_distance(points1, points2):
    # points1: (M,2), points2: (N,2)
    points1 = points1[:, np.newaxis, :]  # (M,1,2)
    points2 = points2[np.newaxis, :, :]  # (1,N,2)
    
    # 计算欧氏距离
    dist = np.sqrt(np.sum((points1 - points2)**2, axis=2))  # (M,N)
    return dist

def batch_points_in_box(points, boxes):
    """
    检查每个点是否在对应的框内
    points: (M,2) - M个点的坐标
    boxes: (M,4) - M个框的坐标 [x1,y1,x2,y2]
    返回: (M,) 布尔数组
    """
    x_check = (points[:,0] >= boxes[:,0]) & (points[:,0] <= boxes[:,2])
    y_check = (points[:,1] >= boxes[:,1]) & (points[:,1] <= boxes[:,3])
    return x_check & y_check

if __name__ == "__main__":
    predict_str = """
<answer>
[{"bbox_2d": [10, 100, 398, 423], "point_2d": [283, 169]}]
</answer>
"""
    ground_truth = """
[{"bbox_2d": [416, 7, 833, 553], "point_2d": [648, 249]}]"""
    print(predict_str)
    print(ground_truth)
    print(vision_reasoner_compute_score(predict_str, ground_truth))
    