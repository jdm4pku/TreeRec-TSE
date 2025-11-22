# 统计每个生态的制品数量（从candidate_artifacts.json文件）
import json
import os
from pathlib import Path

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def count_artifacts_by_ecosystem(data_dir):
    """
    统计每个生态包含的制品数量（从candidate_artifacts.json文件）
    
    Args:
        data_dir: 数据目录路径
    """
    data_path = Path(data_dir)
    statistics = {}
    
    # 遍历数据目录下的所有子目录
    for ecosystem_dir in data_path.iterdir():
        if ecosystem_dir.is_dir():
            ecosystem_name = ecosystem_dir.name
            candidate_file = ecosystem_dir / "candidate_artifacts.json"
            
            # 检查candidate_artifacts.json文件是否存在
            if candidate_file.exists():
                try:
                    data = load_json(candidate_file)
                    
                    # 统计制品数量（JSON数组的长度）
                    artifact_count = len(data)
                    statistics[ecosystem_name] = artifact_count
                    print(f"{ecosystem_name}: {artifact_count} 个制品")
                    
                    # 如果是js或hf生态，还可以按type统计
                    if ecosystem_name in ["js", "hf"] and artifact_count > 0:
                        # 检查第一个元素是否有type字段
                        if isinstance(data[0], dict) and "type" in data[0]:
                            type_counts = {}
                            for item in data:
                                item_type = item.get("type", "unknown")
                                type_counts[item_type] = type_counts.get(item_type, 0) + 1
                            
                            # 显示前10个最常见的type
                            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
                            print(f"  前10个最常见的类型:")
                            for item_type, count in sorted_types[:10]:
                                print(f"    - {item_type}: {count} 个")
                            
                except json.JSONDecodeError as e:
                    print(f"错误: 无法解析 {candidate_file} - {e}")
                except Exception as e:
                    print(f"错误: 读取 {candidate_file} 时出错 - {e}")
            else:
                print(f"警告: {ecosystem_name} 生态中未找到 candidate_artifacts.json 文件")
    
    # 打印汇总信息
    print("\n" + "="*60)
    print("统计汇总:")
    print("="*60)
    total_artifacts = sum(statistics.values())
    for ecosystem, count in sorted(statistics.items()):
        percentage = (count / total_artifacts * 100) if total_artifacts > 0 else 0
        print(f"{ecosystem:15s}: {count:6d} 个制品 ({percentage:5.2f}%)")
    print("="*60)
    print(f"{'总计':15s}: {total_artifacts:6d} 个制品")
    print("="*60)
    
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
        print("统计每个生态的制品数量（从candidate_artifacts.json文件）\n")
        count_artifacts_by_ecosystem(data_dir)

