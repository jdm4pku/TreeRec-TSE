# 统计每个生态中数据集样本的intent长度和artifact长度的平均值、最小值和最大值
import json
import os
from pathlib import Path
import statistics

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def calculate_length_stats(lengths):
    """
    计算长度统计信息
    
    Args:
        lengths: 长度列表
    
    Returns:
        dict: 包含平均值、最小值、最大值的字典
    """
    if not lengths:
        return {
            "mean": 0,
            "min": 0,
            "max": 0,
            "count": 0
        }
    
    return {
        "mean": statistics.mean(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "count": len(lengths)
    }

def analyze_dataset_lengths(data_dir):
    """
    分析每个生态数据集中intent和artifact的长度统计
    
    Args:
        data_dir: 数据目录路径
    """
    data_path = Path(data_dir)
    all_statistics = {}
    
    print(f"正在分析数据目录: {data_path.absolute()}\n")
    print("="*80)
    
    # 遍历数据目录下的所有子目录
    for ecosystem_dir in sorted(data_path.iterdir()):
        if ecosystem_dir.is_dir() and ecosystem_dir.name != "baselines":
            ecosystem_name = ecosystem_dir.name
            dataset_file = ecosystem_dir / "dataset.json"
            
            # 检查dataset.json文件是否存在
            if dataset_file.exists():
                try:
                    dataset = load_json(dataset_file)
                    
                    # 收集intent和artifact的长度
                    intent_lengths = []
                    artifact_lengths = []
                    
                    for sample in dataset:
                        intent = sample.get("intent", "")
                        artifact = sample.get("artifact", "")
                        
                        # 计算字符长度
                        intent_lengths.append(len(intent))
                        artifact_lengths.append(len(artifact))
                    
                    # 计算统计信息
                    intent_stats = calculate_length_stats(intent_lengths)
                    artifact_stats = calculate_length_stats(artifact_lengths)
                    
                    all_statistics[ecosystem_name] = {
                        "intent": intent_stats,
                        "artifact": artifact_stats,
                        "total_samples": len(dataset)
                    }
                    
                    # 打印统计信息
                    print(f"\n生态: {ecosystem_name}")
                    print(f"  总样本数: {len(dataset)}")
                    print(f"\n  Intent长度统计:")
                    print(f"    平均值: {intent_stats['mean']:.2f} 字符")
                    print(f"    最小值: {intent_stats['min']} 字符")
                    print(f"    最大值: {intent_stats['max']} 字符")
                    print(f"\n  Artifact长度统计:")
                    print(f"    平均值: {artifact_stats['mean']:.2f} 字符")
                    print(f"    最小值: {artifact_stats['min']} 字符")
                    print(f"    最大值: {artifact_stats['max']} 字符")
                    print("-"*80)
                    
                except json.JSONDecodeError as e:
                    print(f"错误: 无法解析 {dataset_file} - {e}")
                except Exception as e:
                    print(f"错误: 读取 {dataset_file} 时出错 - {e}")
            else:
                print(f"警告: {ecosystem_name} 生态中未找到 dataset.json 文件")
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("汇总统计表")
    print("="*80)
    print(f"{'生态':<15} {'样本数':<10} {'Intent平均':<15} {'Intent最小':<15} {'Intent最大':<15} {'Artifact平均':<15} {'Artifact最小':<15} {'Artifact最大':<15}")
    print("-"*80)
    
    for ecosystem_name, stats in sorted(all_statistics.items()):
        intent = stats["intent"]
        artifact = stats["artifact"]
        print(f"{ecosystem_name:<15} {stats['total_samples']:<10} "
              f"{intent['mean']:<15.2f} {intent['min']:<15} {intent['max']:<15} "
              f"{artifact['mean']:<15.2f} {artifact['min']:<15} {artifact['max']:<15}")
    
    print("="*80)
    
    # 计算总体统计（加权平均）
    total_samples_all = sum(s["total_samples"] for s in all_statistics.values())
    if total_samples_all > 0:
        # 计算加权平均值
        weighted_intent_mean = sum(s["intent"]["mean"] * s["total_samples"] for s in all_statistics.values()) / total_samples_all
        weighted_artifact_mean = sum(s["artifact"]["mean"] * s["total_samples"] for s in all_statistics.values()) / total_samples_all
        
        overall_intent_min = min(s["intent"]["min"] for s in all_statistics.values())
        overall_intent_max = max(s["intent"]["max"] for s in all_statistics.values())
        overall_artifact_min = min(s["artifact"]["min"] for s in all_statistics.values())
        overall_artifact_max = max(s["artifact"]["max"] for s in all_statistics.values())
        
        print(f"\n{'总计':<15} {total_samples_all:<10} "
              f"{weighted_intent_mean:<15.2f} {overall_intent_min:<15} {overall_intent_max:<15} "
              f"{weighted_artifact_mean:<15.2f} {overall_artifact_min:<15} {overall_artifact_max:<15}")
        print("="*80)
    
    return all_statistics

if __name__ == "__main__":
    # 获取当前脚本所在目录的父目录，然后定位到data目录
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data"
    
    # 如果data_dir不存在，尝试使用相对路径
    if not data_dir.exists():
        data_dir = Path("data")
    
    if not data_dir.exists():
        print(f"错误: 找不到数据目录 {data_dir}")
        print("请确保数据目录存在，或修改脚本中的数据目录路径")
    else:
        analyze_dataset_lengths(data_dir)

