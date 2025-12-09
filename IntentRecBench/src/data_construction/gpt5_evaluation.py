# 使用GPT-5评估每个生态的数据集样本可靠性，并计算与GPT-4o评估结果的一致率
import json
import os
import time
import random
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# 评估prompt模板（与gpt_evaluation.py相同）
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

def gpt5_evaluation(prompt, max_retries=5, retry_delay=1):
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
    # 注意：请通过环境变量设置 OPENAI_API_KEY 和 OPENAI_BASE_URL
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if not api_key:
        raise ValueError("请设置环境变量 OPENAI_API_KEY")
    if not base_url:
        raise ValueError("请设置环境变量 OPENAI_BASE_URL")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # 使用GPT-5模型
    model_name = "gpt-4.1-2025-04-14" # 默认使用gpt-5
    
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

def load_gpt4o_results(output_dir, ecosystem_name):
    """
    加载GPT-4o的评估结果
    
    Args:
        output_dir: 评估结果目录
        ecosystem_name: 生态名称
    
    Returns:
        dict: 以 (intent, artifact) 为键的字典，值为评估结果
    """
    result_file = os.path.join(output_dir, f"{ecosystem_name}_evaluation_results.json")
    if not os.path.exists(result_file):
        return {}
    
    results = load_json(result_file)
    # 创建以 (intent, artifact) 为键的字典
    gpt4o_dict = {}
    for item in results:
        key = (item.get("intent", ""), item.get("artifact", ""))
        gpt4o_dict[key] = item.get("reliable", False)
    
    return gpt4o_dict

def evaluate_ecosystem_with_gpt5(ecosystem_name, dataset_path, gpt4o_output_dir, gpt5_output_dir, sample_ratio=0.1, seed=42):
    """
    使用GPT-5评估单个生态的数据集（采样10%）
    
    Args:
        ecosystem_name: 生态名称
        dataset_path: 数据集文件路径
        gpt4o_output_dir: GPT-4o评估结果目录
        gpt5_output_dir: GPT-5评估结果输出目录
        sample_ratio: 采样比例（默认0.1，即10%）
        seed: 随机种子（确保可复现）
    
    Returns:
        dict: 包含评估统计信息的字典
    """
    print(f"\n{'='*60}")
    print(f"使用GPT-5评估生态: {ecosystem_name}")
    print(f"{'='*60}")
    
    # 加载数据集
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集文件 {dataset_path}")
        return None
    
    dataset = load_json(dataset_path)
    total_samples = len(dataset)
    
    # 采样10%
    random.seed(seed)
    sampled_indices = random.sample(range(total_samples), int(total_samples * sample_ratio))
    sampled_dataset = [dataset[i] for i in sorted(sampled_indices)]
    sampled_count = len(sampled_dataset)
    
    print(f"总样本数: {total_samples}")
    print(f"采样数量: {sampled_count} ({sample_ratio*100:.1f}%)")
    
    # 加载GPT-4o的评估结果用于计算一致率
    gpt4o_results = load_gpt4o_results(gpt4o_output_dir, ecosystem_name)
    print(f"已加载GPT-4o评估结果: {len(gpt4o_results)} 个样本")
    
    # 评估结果
    evaluation_results = []
    reliable_count = 0
    agreement_count = 0  # 与GPT-4o一致的数量
    compared_count = 0   # 可以比较的样本数量（在GPT-4o结果中存在）
    
    # 逐个评估样本
    for idx, sample in enumerate(tqdm(sampled_dataset, desc=f"GPT-5评估 {ecosystem_name}")):
        intent = sample.get("intent", "")
        artifact = sample.get("artifact", "")
        
        # 构建评估prompt
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            intent=intent,
            artifact=artifact,
            ecosystem=ecosystem_name
        )
        
        # 调用GPT-5评估
        result = gpt5_evaluation(prompt)
        is_reliable = (result == "可靠")
        
        if is_reliable:
            reliable_count += 1
        
        # 检查与GPT-4o结果的一致性
        key = (intent, artifact)
        gpt4o_reliable = gpt4o_results.get(key, None)
        agreement = None
        
        if gpt4o_reliable is not None:
            compared_count += 1
            if gpt4o_reliable == is_reliable:
                agreement_count += 1
                agreement = True
            else:
                agreement = False
        
        # 保存详细评估结果
        evaluation_results.append({
            "intent": intent,
            "artifact": artifact,
            "reliable": is_reliable,
            "evaluation_result": result,
            "gpt4o_reliable": gpt4o_reliable,  # None表示GPT-4o结果中不存在
            "agreement": agreement  # None表示无法比较，True/False表示一致/不一致
        })
        
        # 每评估50个样本，打印一次进度
        if (idx + 1) % 50 == 0:
            current_rate = reliable_count / (idx + 1) * 100
            if compared_count > 0:
                current_agreement = agreement_count / compared_count * 100
                print(f"  进度: {idx + 1}/{sampled_count}, 接收率: {current_rate:.2f}%, 一致率: {current_agreement:.2f}%")
            else:
                print(f"  进度: {idx + 1}/{sampled_count}, 接收率: {current_rate:.2f}%")
    
    # 计算统计指标
    acceptance_rate = (reliable_count / sampled_count * 100) if sampled_count > 0 else 0
    agreement_rate = (agreement_count / compared_count * 100) if compared_count > 0 else None
    
    # 保存详细评估结果
    if gpt5_output_dir:
        os.makedirs(gpt5_output_dir, exist_ok=True)
        output_file = os.path.join(gpt5_output_dir, f"{ecosystem_name}_gpt5_evaluation_results.json")
        save_json(evaluation_results, output_file)
        print(f"\n详细评估结果已保存到: {output_file}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"生态 {ecosystem_name} GPT-5评估结果:")
    print(f"{'='*60}")
    print(f"总样本数: {total_samples}")
    print(f"采样数量: {sampled_count} ({sample_ratio*100:.1f}%)")
    print(f"可靠样本数: {reliable_count}")
    print(f"不可靠样本数: {sampled_count - reliable_count}")
    print(f"接收率: {acceptance_rate:.2f}%")
    print(f"\n与GPT-4o结果比较:")
    print(f"  可比较样本数: {compared_count}")
    print(f"  一致样本数: {agreement_count}")
    if agreement_rate is not None:
        print(f"  一致率: {agreement_rate:.2f}%")
    else:
        print(f"  一致率: N/A (无可比较样本)")
    print(f"{'='*60}\n")
    
    return {
        "ecosystem": ecosystem_name,
        "total_samples": total_samples,
        "sampled_samples": sampled_count,
        "reliable_samples": reliable_count,
        "unreliable_samples": sampled_count - reliable_count,
        "acceptance_rate": acceptance_rate,
        "compared_samples": compared_count,
        "agreement_samples": agreement_count,
        "agreement_rate": agreement_rate,
        "evaluation_results": evaluation_results
    }

def evaluate_all_ecosystems_with_gpt5(data_dir, gpt4o_output_dir, gpt5_output_dir, sample_ratio=0.1):
    """
    使用GPT-5评估所有生态的数据集
    
    Args:
        data_dir: 数据目录路径
        gpt4o_output_dir: GPT-4o评估结果目录
        gpt5_output_dir: GPT-5评估结果输出目录
        sample_ratio: 采样比例（默认0.1，即10%）
    
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
            result = evaluate_ecosystem_with_gpt5(
                ecosystem_name, 
                str(dataset_file), 
                gpt4o_output_dir,
                gpt5_output_dir,
                sample_ratio
            )
            if result:
                all_results[ecosystem_name] = result
        else:
            print(f"警告: {ecosystem_name} 生态中未找到 dataset.json 文件")
    
    # 计算总体统计
    total_samples_all = sum(r["total_samples"] for r in all_results.values())
    total_sampled_all = sum(r["sampled_samples"] for r in all_results.values())
    total_reliable_all = sum(r["reliable_samples"] for r in all_results.values())
    overall_acceptance_rate = (total_reliable_all / total_sampled_all * 100) if total_sampled_all > 0 else 0
    
    total_compared_all = sum(r["compared_samples"] for r in all_results.values())
    total_agreement_all = sum(r["agreement_samples"] for r in all_results.values())
    overall_agreement_rate = (total_agreement_all / total_compared_all * 100) if total_compared_all > 0 else None
    
    # 打印总体统计
    print("\n" + "="*60)
    print("GPT-5评估结果汇总")
    print("="*60)
    print(f"{'生态':<15} {'总样本':<10} {'采样':<10} {'可靠':<10} {'接收率':<10} {'可比较':<10} {'一致':<10} {'一致率':<10}")
    print("-"*60)
    
    for ecosystem_name, result in sorted(all_results.items()):
        agreement_str = f"{result['agreement_rate']:.2f}%" if result['agreement_rate'] is not None else "N/A"
        print(f"{ecosystem_name:<15} {result['total_samples']:<10} {result['sampled_samples']:<10} "
              f"{result['reliable_samples']:<10} {result['acceptance_rate']:<10.2f}% "
              f"{result['compared_samples']:<10} {result['agreement_samples']:<10} {agreement_str}")
    
    print("-"*60)
    agreement_str = f"{overall_agreement_rate:.2f}%" if overall_agreement_rate is not None else "N/A"
    print(f"{'总计':<15} {total_samples_all:<10} {total_sampled_all:<10} {total_reliable_all:<10} "
          f"{overall_acceptance_rate:<10.2f}% {total_compared_all:<10} {total_agreement_all:<10} {agreement_str}")
    print("="*60)
    
    # 保存汇总结果
    if gpt5_output_dir:
        os.makedirs(gpt5_output_dir, exist_ok=True)
        summary = {
            "overall_statistics": {
                "total_samples": total_samples_all,
                "sampled_samples": total_sampled_all,
                "total_reliable_samples": total_reliable_all,
                "overall_acceptance_rate": overall_acceptance_rate,
                "total_compared_samples": total_compared_all,
                "total_agreement_samples": total_agreement_all,
                "overall_agreement_rate": overall_agreement_rate
            },
            "ecosystem_statistics": {
                name: {
                    "total_samples": r["total_samples"],
                    "sampled_samples": r["sampled_samples"],
                    "reliable_samples": r["reliable_samples"],
                    "acceptance_rate": r["acceptance_rate"],
                    "compared_samples": r["compared_samples"],
                    "agreement_samples": r["agreement_samples"],
                    "agreement_rate": r["agreement_rate"]
                }
                for name, r in all_results.items()
            }
        }
        summary_file = os.path.join(gpt5_output_dir, "gpt5_evaluation_summary.json")
        save_json(summary, summary_file)
        print(f"\n评估汇总结果已保存到: {summary_file}")
    
    return {
        "overall_statistics": {
            "total_samples": total_samples_all,
            "sampled_samples": total_sampled_all,
            "total_reliable_samples": total_reliable_all,
            "overall_acceptance_rate": overall_acceptance_rate,
            "total_compared_samples": total_compared_all,
            "total_agreement_samples": total_agreement_all,
            "overall_agreement_rate": overall_agreement_rate
        },
        "ecosystem_results": all_results
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用GPT-5评估数据集样本可靠性并计算与GPT-4o的一致率")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据目录路径（默认：自动检测）")
    parser.add_argument("--gpt4o_output_dir", type=str, default=None,
                        help="GPT-4o评估结果目录（默认：data/evaluation_results）")
    parser.add_argument("--gpt5_output_dir", type=str, default=None,
                        help="GPT-5评估结果输出目录（默认：data/gpt5_evaluation_results）")
    parser.add_argument("--ecosystem", type=str, default=None,
                        help="只评估指定生态（默认：评估所有生态）")
    parser.add_argument("--model", type=str, default=None,
                        help="GPT-5模型名称（默认：使用环境变量GPT5_MODEL或gpt-5）")
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                        help="采样比例（默认：0.1，即10%）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认：42）")
    
    args = parser.parse_args()
    
    # 设置模型（如果指定）
    if args.model:
        os.environ["GPT5_MODEL"] = args.model
    
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
    if args.gpt4o_output_dir:
        gpt4o_output_dir = args.gpt4o_output_dir
    else:
        gpt4o_output_dir = os.path.join(data_dir, "evaluation_results")
    
    if args.gpt5_output_dir:
        gpt5_output_dir = args.gpt5_output_dir
    else:
        gpt5_output_dir = os.path.join(data_dir, "gpt5_evaluation_results")
    
    print(f"数据目录: {data_dir}")
    print(f"GPT-4o结果目录: {gpt4o_output_dir}")
    print(f"GPT-5结果目录: {gpt5_output_dir}")
    print(f"使用模型: {os.environ.get('GPT5_MODEL', 'gpt-5')}")
    print(f"采样比例: {args.sample_ratio*100:.1f}%")
    print(f"随机种子: {args.seed}")
    
    # 执行评估
    if args.ecosystem:
        # 只评估指定生态
        dataset_path = os.path.join(data_dir, args.ecosystem, "dataset.json")
        if not os.path.exists(dataset_path):
            print(f"错误: 找不到 {args.ecosystem} 生态的数据集文件")
            exit(1)
        evaluate_ecosystem_with_gpt5(
            args.ecosystem, 
            dataset_path, 
            gpt4o_output_dir,
            gpt5_output_dir,
            args.sample_ratio,
            args.seed
        )
    else:
        # 评估所有生态
        evaluate_all_ecosystems_with_gpt5(
            data_dir, 
            gpt4o_output_dir,
            gpt5_output_dir,
            args.sample_ratio
        )

