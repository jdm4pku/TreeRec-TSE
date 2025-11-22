# 使用GPT-5评估每个生态的数据集样本可靠性，并计算接收率
import json
import os
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# 评估prompt模板
EVALUATION_PROMPT_TEMPLATE = """
你是一个数据质量评估专家。请评估以下样本是否可靠。

评估标准：
1. Intent（意图描述）是否清晰、合理、完整
2. Intent和Artifact（工件）是否匹配
3. 样本是否具有实际意义和价值
4. 是否存在明显的错误、矛盾或不合理之处

请仔细分析以下样本，然后只回答"可靠"或"不可靠"，不要添加任何其他内容。

样本：
Intent: {intent}
Artifact: {artifact}
Ecosystem: {ecosystem}

评估结果（只回答"可靠"或"不可靠"）：
"""

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_json(data, file_path):
    """保存JSON文件"""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def gpt_evaluation(prompt, max_retries=5, retry_delay=1):
    """
    调用GPT-5进行样本可靠性评估
    
    Args:
        prompt: 评估prompt
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        str: 评估结果（"可靠"或"不可靠"）
    """
    # 使用与annotation.py相同的API配置
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "http://66.206.9.230:4000/v1"),
    )
    
    model_name = "gpt-4o-2024-05-13"  # 默认使用gpt-4o，可通过环境变量改为gpt-5
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=0  # 使用0温度以获得更稳定的结果
            )
            result = response.choices[0].message.content.strip()
            
            # 标准化结果：只保留"可靠"或"不可靠"
            if "可靠" in result and "不可靠" not in result:
                return "可靠"
            elif "不可靠" in result:
                return "不可靠"
            else:
                # 如果结果不明确，尝试解析
                result_lower = result.lower()
                if "reliable" in result_lower or "yes" in result_lower or "true" in result_lower:
                    return "可靠"
                else:
                    return "不可靠"
                    
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  错误: {e}, 重试中... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay * (attempt + 1))  # 指数退避
            else:
                print(f"  错误: 达到最大重试次数，返回'不可靠'")
                return "不可靠"
    
    return "不可靠"

def evaluate_ecosystem(ecosystem_name, dataset_path, output_dir=None, skip_if_exists=True):
    """
    评估单个生态的数据集
    
    Args:
        ecosystem_name: 生态名称
        dataset_path: 数据集文件路径
        output_dir: 输出目录（用于保存详细评估结果）
        skip_if_exists: 如果结果文件已存在，是否跳过评估直接读取（默认True）
    
    Returns:
        dict: 包含评估统计信息的字典
    """
    print(f"\n{'='*60}")
    print(f"正在处理生态: {ecosystem_name}")
    print(f"{'='*60}")
    
    # 检查是否已有评估结果文件
    if output_dir and skip_if_exists:
        result_file = os.path.join(output_dir, f"{ecosystem_name}_evaluation_results.json")
        if os.path.exists(result_file):
            print(f"✅ 发现已有评估结果文件: {result_file}")
            print("📊 直接读取并统计结果...")
            
            try:
                evaluation_results = load_json(result_file)
                total_samples = len(evaluation_results)
                reliable_count = sum(1 for r in evaluation_results if r.get("reliable", False))
                unreliable_count = total_samples - reliable_count
                acceptance_rate = (reliable_count / total_samples * 100) if total_samples > 0 else 0
                
                # 打印统计信息
                print(f"\n{'='*60}")
                print(f"生态 {ecosystem_name} 评估结果（从已有文件读取）:")
                print(f"{'='*60}")
                print(f"总样本数: {total_samples}")
                print(f"可靠样本数: {reliable_count}")
                print(f"不可靠样本数: {unreliable_count}")
                print(f"接收率: {acceptance_rate:.2f}%")
                print(f"{'='*60}\n")
                
                return {
                    "ecosystem": ecosystem_name,
                    "total_samples": total_samples,
                    "reliable_samples": reliable_count,
                    "unreliable_samples": unreliable_count,
                    "acceptance_rate": acceptance_rate,
                    "evaluation_results": evaluation_results
                }
            except Exception as e:
                print(f"⚠️  读取已有结果文件时出错: {e}")
                print("🔄 将重新进行评估...")
    
    # 如果没有已有结果或读取失败，进行新的评估
    print(f"🔄 开始评估生态: {ecosystem_name}")
    
    # 加载数据集
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集文件 {dataset_path}")
        return None
    
    dataset = load_json(dataset_path)
    total_samples = len(dataset)
    
    print(f"总样本数: {total_samples}")
    
    # 评估结果
    evaluation_results = []
    reliable_count = 0
    
    # 逐个评估样本
    for idx, sample in enumerate(tqdm(dataset, desc=f"评估 {ecosystem_name}")):
        intent = sample.get("intent", "")
        artifact = sample.get("artifact", "")
        
        # 构建评估prompt
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            intent=intent,
            artifact=artifact,
            ecosystem=ecosystem_name
        )
        
        # 调用GPT评估
        result = gpt_evaluation(prompt)
        is_reliable = (result == "可靠")
        
        if is_reliable:
            reliable_count += 1
        
        # 保存详细评估结果
        evaluation_results.append({
            "intent": intent,
            "artifact": artifact,
            "reliable": is_reliable,
            "evaluation_result": result
        })
        
        # 每评估100个样本，打印一次进度
        if (idx + 1) % 100 == 0:
            current_rate = reliable_count / (idx + 1) * 100
            print(f"  进度: {idx + 1}/{total_samples}, 当前接收率: {current_rate:.2f}%")
    
    # 计算接收率
    acceptance_rate = (reliable_count / total_samples * 100) if total_samples > 0 else 0
    
    # 保存详细评估结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{ecosystem_name}_evaluation_results.json")
        save_json(evaluation_results, output_file)
        print(f"\n详细评估结果已保存到: {output_file}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"生态 {ecosystem_name} 评估结果:")
    print(f"{'='*60}")
    print(f"总样本数: {total_samples}")
    print(f"可靠样本数: {reliable_count}")
    print(f"不可靠样本数: {total_samples - reliable_count}")
    print(f"接收率: {acceptance_rate:.2f}%")
    print(f"{'='*60}\n")
    
    return {
        "ecosystem": ecosystem_name,
        "total_samples": total_samples,
        "reliable_samples": reliable_count,
        "unreliable_samples": total_samples - reliable_count,
        "acceptance_rate": acceptance_rate,
        "evaluation_results": evaluation_results
    }

def generate_summary_from_results(output_dir):
    """
    从已有的评估结果文件生成汇总统计
    
    Args:
        output_dir: 输出目录路径（包含评估结果文件）
    
    Returns:
        dict: 汇总统计信息
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"错误: 输出目录不存在: {output_dir}")
        return None
    
    all_results = {}
    
    # 查找所有评估结果文件
    result_files = list(output_path.glob("*_evaluation_results.json"))
    
    if not result_files:
        print(f"警告: 在 {output_dir} 中未找到评估结果文件")
        return None
    
    print(f"找到 {len(result_files)} 个评估结果文件")
    
    # 读取每个生态的评估结果
    for result_file in result_files:
        # 从文件名提取生态名称（例如：js_evaluation_results.json -> js）
        ecosystem_name = result_file.stem.replace("_evaluation_results", "")
        
        try:
            results = load_json(result_file)
            
            # 统计可靠样本数
            total_samples = len(results)
            reliable_samples = sum(1 for r in results if r.get("reliable", False))
            unreliable_samples = total_samples - reliable_samples
            acceptance_rate = (reliable_samples / total_samples * 100) if total_samples > 0 else 0
            
            all_results[ecosystem_name] = {
                "total_samples": total_samples,
                "reliable_samples": reliable_samples,
                "unreliable_samples": unreliable_samples,
                "acceptance_rate": acceptance_rate
            }
            
            print(f"{ecosystem_name}: {total_samples} 个样本, {reliable_samples} 个可靠 ({acceptance_rate:.2f}%)")
            
        except Exception as e:
            print(f"错误: 读取 {result_file} 时出错 - {e}")
            continue
    
    if not all_results:
        print("错误: 未能读取任何评估结果")
        return None
    
    # 计算总体统计
    total_samples_all = sum(r["total_samples"] for r in all_results.values())
    total_reliable_all = sum(r["reliable_samples"] for r in all_results.values())
    overall_acceptance_rate = (total_reliable_all / total_samples_all * 100) if total_samples_all > 0 else 0
    
    # 打印总体统计
    print("\n" + "="*60)
    print("总体评估结果汇总")
    print("="*60)
    print(f"{'生态':<15} {'总样本数':<12} {'可靠样本':<12} {'接收率':<10}")
    print("-"*60)
    
    for ecosystem_name, result in sorted(all_results.items()):
        print(f"{ecosystem_name:<15} {result['total_samples']:<12} {result['reliable_samples']:<12} {result['acceptance_rate']:<10.2f}%")
    
    print("-"*60)
    print(f"{'总计':<15} {total_samples_all:<12} {total_reliable_all:<12} {overall_acceptance_rate:<10.2f}%")
    print("="*60)
    
    # 保存汇总结果
    summary = {
        "overall_statistics": {
            "total_samples": total_samples_all,
            "total_reliable_samples": total_reliable_all,
            "overall_acceptance_rate": overall_acceptance_rate
        },
        "ecosystem_statistics": {
            name: {
                "total_samples": r["total_samples"],
                "reliable_samples": r["reliable_samples"],
                "acceptance_rate": r["acceptance_rate"]
            }
            for name, r in all_results.items()
        }
    }
    summary_file = output_path / "evaluation_summary.json"
    save_json(summary, summary_file)
    print(f"\n评估汇总结果已保存到: {summary_file}")
    
    return summary

def evaluate_all_ecosystems(data_dir, output_dir=None):
    """
    评估所有生态的数据集
    
    Args:
        data_dir: 数据目录路径
        output_dir: 输出目录路径（用于保存评估结果）
    
    Returns:
        dict: 包含所有生态评估结果的字典
    """
    data_path = Path(data_dir)
    all_results = {}
    
    # 遍历数据目录下的所有子目录
    ecosystem_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name != "baselines"]
    
    if not ecosystem_dirs:
        print(f"错误: 在 {data_dir} 中未找到生态目录")
        return None
    
    print(f"找到 {len(ecosystem_dirs)} 个生态: {[d.name for d in ecosystem_dirs]}")
    
    # 评估每个生态
    for ecosystem_dir in ecosystem_dirs:
        ecosystem_name = ecosystem_dir.name
        dataset_file = ecosystem_dir / "dataset.json"
        
        if dataset_file.exists():
            result = evaluate_ecosystem(ecosystem_name, str(dataset_file), output_dir)
            if result:
                all_results[ecosystem_name] = result
        else:
            print(f"警告: {ecosystem_name} 生态中未找到 dataset.json 文件")
    
    # 计算总体统计
    total_samples_all = sum(r["total_samples"] for r in all_results.values())
    total_reliable_all = sum(r["reliable_samples"] for r in all_results.values())
    overall_acceptance_rate = (total_reliable_all / total_samples_all * 100) if total_samples_all > 0 else 0
    
    # 打印总体统计
    print("\n" + "="*60)
    print("总体评估结果汇总")
    print("="*60)
    print(f"{'生态':<15} {'总样本数':<12} {'可靠样本':<12} {'接收率':<10}")
    print("-"*60)
    
    for ecosystem_name, result in sorted(all_results.items()):
        print(f"{ecosystem_name:<15} {result['total_samples']:<12} {result['reliable_samples']:<12} {result['acceptance_rate']:<10.2f}%")
    
    print("-"*60)
    print(f"{'总计':<15} {total_samples_all:<12} {total_reliable_all:<12} {overall_acceptance_rate:<10.2f}%")
    print("="*60)
    
    # 保存汇总结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary = {
            "overall_statistics": {
                "total_samples": total_samples_all,
                "total_reliable_samples": total_reliable_all,
                "overall_acceptance_rate": overall_acceptance_rate
            },
            "ecosystem_statistics": {
                name: {
                    "total_samples": r["total_samples"],
                    "reliable_samples": r["reliable_samples"],
                    "acceptance_rate": r["acceptance_rate"]
                }
                for name, r in all_results.items()
            }
        }
        summary_file = os.path.join(output_dir, "evaluation_summary.json")
        save_json(summary, summary_file)
        print(f"\n评估汇总结果已保存到: {summary_file}")
    
    return {
        "overall_statistics": {
            "total_samples": total_samples_all,
            "total_reliable_samples": total_reliable_all,
            "overall_acceptance_rate": overall_acceptance_rate
        },
        "ecosystem_results": all_results
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用GPT-5评估数据集样本可靠性")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据目录路径（默认：自动检测）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录路径（默认：data/evaluation_results）")
    parser.add_argument("--ecosystem", type=str, default=None,
                        help="只评估指定生态（默认：评估所有生态）")
    parser.add_argument("--model", type=str, default=None,
                        help="GPT模型名称（默认：使用环境变量GPT_MODEL或gpt-4o-2024-05-13）")
    parser.add_argument("--generate_summary", action="store_true",
                        help="从已有的评估结果文件生成汇总统计（不进行评估）")
    
    args = parser.parse_args()
    
    # 设置模型（如果指定）
    if args.model:
        os.environ["GPT_MODEL"] = args.model
    
    # 确定数据目录
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # 自动检测数据目录
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        data_dir = project_root / "data"
        
        if not data_dir.exists():
            data_dir = Path("data")
    
    if not os.path.exists(data_dir):
        print(f"错误: 找不到数据目录 {data_dir}")
        exit(1)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(data_dir, "evaluation_results")
    
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"使用模型: {os.environ.get('GPT_MODEL', 'gpt-4o-2024-05-13')}")
    
    # 如果只是生成汇总，直接调用函数并退出
    if args.generate_summary:
        generate_summary_from_results(output_dir)
        exit(0)
    
    # 执行评估
    if args.ecosystem:
        # 只评估指定生态
        dataset_path = os.path.join(data_dir, args.ecosystem, "dataset.json")
        if not os.path.exists(dataset_path):
            print(f"错误: 找不到 {args.ecosystem} 生态的数据集文件")
            exit(1)
        evaluate_ecosystem(args.ecosystem, dataset_path, output_dir)
    else:
        # 评估所有生态
        evaluate_all_ecosystems(data_dir, output_dir)

