# 统计每个生态的数据集的样本数量
import json
import os
from pathlib import Path

def count_samples_by_ecosystem(data_dir):
    """
    统计每个生态包含的样本数量
    
    Args:
        data_dir: 数据目录路径
    """
    data_path = Path(data_dir)
    statistics = {}
    
    # 遍历数据目录下的所有子目录
    for ecosystem_dir in data_path.iterdir():
        if ecosystem_dir.is_dir():
            ecosystem_name = ecosystem_dir.name
            dataset_file = ecosystem_dir / "dataset.json"
            
            # 检查dataset.json文件是否存在
            if dataset_file.exists():
                try:
                    with open(dataset_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 统计样本数量（JSON数组的长度）
                        sample_count = len(data)
                        statistics[ecosystem_name] = sample_count
                        print(f"{ecosystem_name}: {sample_count} 个样本")
                except json.JSONDecodeError as e:
                    print(f"错误: 无法解析 {dataset_file} - {e}")
                except Exception as e:
                    print(f"错误: 读取 {dataset_file} 时出错 - {e}")
            else:
                print(f"警告: {ecosystem_name} 生态中未找到 dataset.json 文件")
    
    # 打印汇总信息
    print("\n" + "="*50)
    print("统计汇总:")
    print("="*50)
    total_samples = sum(statistics.values())
    for ecosystem, count in sorted(statistics.items()):
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{ecosystem:15s}: {count:5d} 个样本 ({percentage:5.2f}%)")
    print("="*50)
    print(f"{'总计':15s}: {total_samples:5d} 个样本")
    print("="*50)
    
    return statistics

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
        print(f"正在统计数据目录: {data_dir.absolute()}\n")
        count_samples_by_ecosystem(data_dir)
