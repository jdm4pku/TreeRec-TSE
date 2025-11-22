import os
import json
import time
import argparse
import re
from typing import List, Dict, Tuple, Optional
from statistics import mean, pstdev
from math import log2

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # 如果没有tqdm，创建一个简单的进度条替代
    def tqdm(iterable, desc=None, total=None, **kwargs):
        return iterable

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    SentenceTransformer = None

os.environ["OPENAI_API_KEY"] = "sk-BwTI1iSg83soUQ6u2d1096B8A27848E5B3E4141154Dc592b"
os.environ["OPENAI_BASE_URL"] = "https://api.yesapikey.com/v1"

# ========== 工具函数 ==========

def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, (list, dict)):
        raise ValueError(f"Invalid JSON format in {file_path}")
    return data


def save_json(data, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ========== 评估指标 ==========

def precision_at_k(ranked_names: List[str], gold: str, k: int) -> float:
    """P@K as Hit@K: 1 if gold appears in top-K, else 0."""
    top_k = ranked_names[:k]
    return 1.0 if gold in top_k else 0.0


def dcg_at_k(ranked_names: List[str], gold: str, k: int) -> float:
    """计算 DCG@K（二元相关性）"""
    for idx, name in enumerate(ranked_names[:k], start=1):
        if name == gold:
            return 1.0 / log2(1.0 + idx)
    return 0.0


# ========== LLM模型配置 ==========

def get_model_config(model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    """
    根据模型名称获取对应的API配置
    支持: GPT, Qwen, DeepSeek, Llama
    """
    model_lower = model_name.lower()
    
    # 如果用户明确指定了 base_url 和 api_key，优先使用
    if base_url and api_key:
        return {
            "base_url": base_url,
            "api_key": api_key,
            "model_name": model_name
        }
    
    # GPT 系列模型（使用 OpenAI 官方 API）
    print("***************")
    print(f"model_lower: {model_lower}")
    print("***************")
    if "gpt" in model_lower or "openai" in model_lower:
        return {
            "base_url": base_url or "http://66.206.9.230:4000/v1",
            "api_key": api_key or "sk-BwTI1iSg83soUQ6u2d1096B8A27848E5B3E4141154Dc592b",
            "model_name": model_name
        }
    
    print("***************")
    print(f"model_lower: {model_lower}")
    print("***************")
    # Qwen 系列模型（使用 SiliconFlow）
    if "qwen" in model_lower:
        return {
            "base_url": base_url or "https://api.siliconflow.cn/v1",
            "api_key": api_key or os.environ.get("SILICONFLOW_API_KEY", 
                        "sk-wbnxvocaaofhilzlgkvhiuhoivdawabyvaavkvblnokomdyz"),
            "model_name": model_name
        }
    
    # DeepSeek 系列模型（使用 SiliconFlow）
    print("***************")
    print(f"model_lower: {model_lower}")
    print("***************")
    if "deepseek" in model_lower:
        return {
            "base_url": base_url or "https://api.siliconflow.cn/v1",
            "api_key": api_key or os.environ.get("SILICONFLOW_API_KEY",
                        "sk-wbnxvocaaofhilzlgkvhiuhoivdawabyvaavkvblnokomdyz"),
            "model_name": model_name
        }
    
    # Llama 系列模型（使用 OpenRouter）
    if "llama" in model_lower:
        return {
            "base_url": base_url or "https://openrouter.ai/api/v1",
            "api_key": api_key or os.environ.get("OPENROUTER_API_KEY",
                        "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"),
            "model_name": model_name
        }
    
    # 抛出错误
    raise ValueError(f"Unsupported model: {model_name}")
    # # 默认使用环境变量或用户指定的配置
    # return {
    #     "base_url": base_url or os.environ.get("OPENAI_BASE_URL", None),
    #     "api_key": api_key or os.environ.get("OPENAI_API_KEY", None),
    #     "model_name": model_name
    # }


def create_llm_client(model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    """
    根据模型名称创建对应的 OpenAI 客户端
    """
    if OpenAI is None:
        raise ImportError("请先安装 openai：pip install openai")
    
    config = get_model_config(model_name, api_key, base_url)
    
    client_kwargs = {}
    if config["api_key"]:
        client_kwargs["api_key"] = config["api_key"]
    if config["base_url"]:
        client_kwargs["base_url"] = config["base_url"]
    
    # 增加超时时间，避免 Connection error（默认 120 秒）
    # timeout 参数格式: (connect_timeout, read_timeout)
    client_kwargs["timeout"] = 120.0  # 总超时时间 120 秒
    
    return OpenAI(**client_kwargs), config["model_name"]


# ========== LLM推荐核心模块 ==========

def build_candidate_text(candidates: List[Dict], max_candidates: int = None, ecosystem: str = None) -> Tuple[List[str], str]:
    """
    构建候选制品的文本描述
    返回: (候选制品名称列表, 格式化的候选制品文本)
    """
    candidate_names = []
    candidate_texts = []
    
    candidates_to_use = candidates[:max_candidates] if max_candidates else candidates
    
    for idx, item in enumerate(candidates_to_use, 1):
        name = item.get("name", "").strip()
        description = item.get("description", "").strip()
        
        candidate_names.append(name)
        
        # 构建候选制品的描述文本（只包含Name和Description）
        text = f"{idx}. Name: {name}\n"
        if description:
            # 对于hf生态，限制description长度为1000字符（参考tree_structures.py）
            if ecosystem == "hf":
                max_desc_len = 1000
                desc = description[:max_desc_len] if len(description) > max_desc_len else description
                if len(description) > max_desc_len:
                    desc += " ... (truncated)"
            else:
                # 其他生态使用全部description，不截断
                desc = description
            text += f"   Description: {desc}\n"
        candidate_texts.append(text)
    
    formatted_text = "\n".join(candidate_texts)
    return candidate_names, formatted_text


def build_scoring_prompt(intent: str, artifact_name: str, artifact_type: str, 
                        artifact_description: str, ecosystem: str) -> str:
    """构建单个制品相关性打分的提示词"""
    ecosystem_names = {
        "hf": "Hugging Face",
        "js": "npm/JavaScript",
        "linux": "Linux"
    }
    ecosystem_name = ecosystem_names.get(ecosystem, ecosystem)
    
    # 对于hf生态，限制description长度为1000字符（参考tree_structures.py）
    if ecosystem == "hf" and artifact_description:
        max_desc_len = 1000
        desc_text = artifact_description[:max_desc_len]
        if len(artifact_description) > max_desc_len:
            desc_text += " ... (truncated)"
    else:
        # 其他生态使用全部description，不截断
        desc_text = artifact_description if artifact_description else "No description"
    
    prompt = f"""You are an expert in {ecosystem_name} ecosystem artifact recommendation. Please evaluate the semantic relevance between the following artifact and the user intent.

User Intent:
{intent}

Artifact Information:
- Name: {artifact_name}
- Description: {desc_text}

Please provide a semantic relevance score (0-100 integer, where 100 indicates a perfect match and 0 indicates complete irrelevance) between this artifact and the user intent.
Return only a number, without any additional text."""
    
    return prompt


def build_batch_scoring_prompt(intent: str, artifacts: List[Dict], ecosystem: str) -> str:
    """构建批量制品相关性打分的提示词"""
    ecosystem_names = {
        "hf": "Hugging Face",
        "js": "npm/JavaScript",
        "linux": "Linux"
    }
    ecosystem_name = ecosystem_names.get(ecosystem, ecosystem)
    
    artifacts_text = []
    for idx, artifact in enumerate(artifacts, 1):
        name = artifact.get("name", "").strip()
        artifact_type = artifact.get("type", "").strip()
        description = artifact.get("description", "").strip()
        
        # 对于hf生态，限制description长度为1000字符
        if ecosystem == "hf" and description:
            max_desc_len = 500
            desc_text = description[:max_desc_len]
            if len(description) > max_desc_len:
                desc_text += " ... (truncated)"
        else:
            desc_text = description if description else "No description"
        
        artifacts_text.append(f"{idx}. Name: {name}\n   Description: {desc_text}")
    
    artifacts_list = "\n\n".join(artifacts_text)
    
    prompt = f"""You are an expert in {ecosystem_name} ecosystem artifact recommendation. Please evaluate the semantic relevance between each of the following artifacts and the user intent.

User Intent:
{intent}

Artifact List:
{artifacts_list}

Please provide a semantic relevance score (0-100 integer) for each artifact, where 100 indicates a perfect match and 0 indicates complete irrelevance.
Return the scores in the following format (one score per line, in the same order as the artifacts):
score1
score2
score3
...

Return only the numbers, one per line, without any additional text."""
    
    return prompt


def build_llm_prompt(intent: str, candidate_text: str, ecosystem: str, top_k: int) -> str:
    """构建LLM推荐提示词"""
    ecosystem_names = {
        "hf": "Hugging Face",
        "js": "npm/JavaScript",
        "linux": "Linux"
    }
    ecosystem_name = ecosystem_names.get(ecosystem, ecosystem)
    
    prompt = f"""You are an expert in {ecosystem_name} ecosystem artifact recommendation. Based on the user's intent, recommend the most relevant artifacts from the given candidate list.

User Intent:
{intent}

Candidate Artifact List:
{candidate_text}

Please select the top {top_k} most relevant artifacts from the above candidate list based on the user intent, ranked from highest to lowest relevance.
Return only the artifact names (the Name field), one per line, without any numbering, prefixes, or additional text.
Example format:
artifact_name1
artifact_name2
artifact_name3

Please directly output the recommended artifact names:"""
    
    return prompt


def _call_llm_api(client: OpenAI, model_name: str, prompt: str, system_content: str = None, max_tokens: int = 2000):
    """调用LLM API，带重试机制"""
    max_retries = 10  # 增加重试次数，特别是对于 Connection error
    default_system = "You are a professional artifact recommendation assistant capable of accurately understanding user intents and recommending the most relevant artifacts."
    system_content = system_content or default_system
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 降低温度以获得更稳定的结果
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e).lower()
            is_connection_error = any(keyword in error_str for keyword in [
                "connection", "timeout", "network", "connect", "refused", 
                "unreachable", "reset", "broken pipe"
            ])
            
            if attempt < max_retries - 1:
                # 对于连接错误，使用更长的等待时间
                if is_connection_error:
                    wait_time = min(5 * (attempt + 1), 60)  # 连接错误：5秒起步，最多60秒
                    print(f"⚠️ 连接错误 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__}, 等待 {wait_time} 秒后重试...")
                else:
                    wait_time = min(2 ** attempt, 20)  # 其他错误：指数退避，最多20秒
                    print(f"⚠️ API调用错误 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__}, 等待 {wait_time} 秒后重试...")
                
                time.sleep(wait_time)
                continue
            else:
                # 最后一次尝试失败，打印详细错误信息
                print(f"❌ API调用最终失败: {type(e).__name__}: {e}")
                raise e


def score_artifact_relevance(intent: str, artifact: Dict, ecosystem: str,
                            client: OpenAI, model_name: str) -> float:
    """
    使用LLM对单个制品进行语义相关性打分
    
    In the scoring stage, each artifact is individually paired with the intent 
    and evaluated by the LLM, which assigns a semantic relevance score on a 0–100 scale
    —where 0 indicates complete irrelevance and 100 denotes a perfect match.
    
    返回: 语义相关性分数 (0-100)
    """
    name = artifact.get("name", "").strip()
    artifact_type = artifact.get("type", "").strip()
    description = artifact.get("description", "").strip()
    
    prompt = build_scoring_prompt(intent, name, artifact_type, description, ecosystem)
    
    try:
        content = _call_llm_api(client, model_name, prompt, 
                                system_content="You are a professional artifact semantic relevance evaluation assistant capable of accurately assessing the semantic relevance between artifacts and user intents.",
                                max_tokens=50)
        
        # 解析分数（尝试提取数字）
        numbers = re.findall(r'\d+', content)
        if numbers:
            score = float(numbers[0])
            # 确保分数在0-100范围内
            score = max(0, min(100, score))
            return score
        else:
            # 如果无法解析，返回默认分数
            return 50.0  # 0-100区间的中位数
    except Exception as e:
        print(f"⚠️ 打分出错 ({name}): {e}")
        return 0.0  # 出错时返回0分


def score_artifacts_batch(intent: str, artifacts: List[Dict], ecosystem: str,
                         client: OpenAI, model_name: str) -> List[float]:
    """
    使用LLM对一批制品进行批量语义相关性打分（优化性能）
    
    返回: 语义相关性分数列表 (0-100)
    """
    if not artifacts:
        return []
    
    prompt = build_batch_scoring_prompt(intent, artifacts, ecosystem)
    
    try:
        # 根据批量大小调整max_tokens（每个制品约50 tokens）
        max_tokens = max(200, len(artifacts) * 50)
        content = _call_llm_api(client, model_name, prompt, 
                                system_content="You are a professional artifact semantic relevance evaluation assistant capable of accurately assessing the semantic relevance between artifacts and user intents.",
                                max_tokens=max_tokens)
        
        # 解析分数（按行提取数字）
        scores = []
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 提取第一个数字
            numbers = re.findall(r'\d+', line)
            if numbers:
                score = float(numbers[0])
                score = max(0, min(100, score))  # 确保在0-100范围内
                scores.append(score)
        
        # 如果解析出的分数数量不足，用默认分数填充
        while len(scores) < len(artifacts):
            scores.append(50.0)  # 使用中位数作为默认分数
        
        # 只返回前len(artifacts)个分数
        return scores[:len(artifacts)]
        
    except Exception as e:
        error_str = str(e).lower()
        is_connection_error = any(keyword in error_str for keyword in [
            "connection", "timeout", "network", "connect", "refused", 
            "unreachable", "reset", "broken pipe"
        ])
        
        if is_connection_error:
            print(f"⚠️ 批量打分连接错误: {type(e).__name__}: {e}")
            print(f"   返回默认分数列表 ({len(artifacts)} 个制品)")
        else:
            print(f"⚠️ 批量打分出错: {type(e).__name__}: {e}")
        
        # 如果出错，返回默认分数列表
        return [50.0] * len(artifacts)


def filter_candidates_by_scoring(intent: str, candidates: List[Dict], ecosystem: str,
                                 client: OpenAI, model_name: str, 
                                 top_percent: float = 0.1, batch_size: int = 20) -> List[Dict]:
    """
    第一阶段：使用LLM对每个制品打分，筛选出top百分比的最相关候选制品
    
    In the scoring stage, each artifact is individually paired with the intent 
    and evaluated by the LLM, which assigns a semantic relevance score on a 0–100 scale.
    This stage produces a numeric relevance distribution over all artifacts.
    
    Args:
        intent: 用户意图
        candidates: 所有候选制品列表
        ecosystem: 生态系统名称
        client: LLM客户端
        model_name: 模型名称
        top_percent: 筛选出的候选制品百分比（默认0.1，即top 10%）
        batch_size: 批处理大小（默认20，即每次批量打分20个制品）
    
    Returns:
        筛选后的候选制品列表（按分数降序排列）
    """
    print(f"📊 第一阶段（Scoring Stage）：对 {len(candidates)} 个候选制品进行语义相关性打分...")
    print(f"   将选择 top {top_percent*100:.1f}% 的候选制品进入推荐阶段")
    print(f"   使用批量打分模式，批量大小: {batch_size}")
    
    scored_candidates = []
    total = len(candidates)
    
    # 将候选制品分批处理
    num_batches = (total + batch_size - 1) // batch_size
    
    # 使用进度条显示打分进度
    if TQDM_AVAILABLE:
        batch_iter = tqdm(range(num_batches), desc="Scoring batches", unit="batch", 
                          total=num_batches, ncols=100)
    else:
        batch_iter = range(num_batches)
    
    for batch_idx in batch_iter:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_candidates = candidates[start_idx:end_idx]
        
        # 批量打分
        batch_scores = score_artifacts_batch(intent, batch_candidates, ecosystem, client, model_name)
        
        # 将分数和候选制品配对
        for score, artifact in zip(batch_scores, batch_candidates):
            scored_candidates.append((score, artifact))
        
        # 添加延迟以避免API限流（对于 qwen 模型，使用更长的延迟以避免连接错误）
        if batch_idx < num_batches - 1:  # 最后一批不需要延迟
            # 检查是否是 qwen 模型，如果是则使用更长的延迟
            model_lower = model_name.lower()
            if "qwen" in model_lower:
                time.sleep(1.0)  # qwen 模型延迟 1 秒
            else:
                time.sleep(0.1)  # 其他模型延迟 0.1 秒
    
    # 按分数降序排序
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 计算要选择的候选数量（至少选择1个）
    top_n = max(1, int(len(candidates) * top_percent))
    
    # 返回top百分比个候选制品
    top_candidates = [artifact for score, artifact in scored_candidates[:top_n]]
    top_scores = [score for score, artifact in scored_candidates[:top_n]]
    
    print(f"✅ 筛选完成，选出 {len(top_candidates)} 个候选制品（分数范围: {min(top_scores):.1f} - {max(top_scores):.1f}）")
    
    return top_candidates


def get_llm_recommendations(intent: str, candidates: List[Dict], ecosystem: str, 
                            top_k: int, client: OpenAI, model_name: str,
                            max_candidates: int = None, use_two_stage: bool = True,
                            filter_top_percent: float = 0.1, scoring_batch_size: int = 20) -> List[str]:
    """
    使用LLM获取推荐结果（两阶段策略）
    
    In the scoring stage, each artifact is individually paired with the intent 
    and evaluated by the LLM, which assigns a semantic relevance score on a 0–100 scale.
    This stage produces a numeric relevance distribution over all artifacts.
    
    In the recommendation stage, we select the top-scored subset (typically the top 10%) 
    as candidates and prompt the LLM again to generate the final top-k recommendations 
    through comparative reasoning over the selected subset.
    
    Args:
        intent: 用户意图
        candidates: 候选制品列表
        ecosystem: 生态系统名称
        top_k: 最终返回的推荐数量
        client: LLM客户端
        model_name: 模型名称
        max_candidates: 最大候选数量（已废弃，保留以兼容）
        use_two_stage: 是否使用两阶段策略（先打分筛选，再推荐）
        filter_top_percent: 第一阶段筛选出的候选百分比（默认0.1，即top 10%）
        scoring_batch_size: 批量打分的大小（默认20）
    
    Returns:
        推荐的制品名称列表（按相关性排序）
    """
    # 如果候选数量较少，直接使用单阶段推荐
    min_candidates_for_two_stage = max(10, int(1 / filter_top_percent))  # 至少需要能选出1个候选
    if not use_two_stage or len(candidates) <= min_candidates_for_two_stage:
        return _get_llm_recommendations_single_stage(intent, candidates, ecosystem, 
                                                   top_k, client, model_name)
    
    # 两阶段策略
    # 第一阶段：打分筛选（Scoring Stage）
    filtered_candidates = filter_candidates_by_scoring(
        intent, candidates, ecosystem, client, model_name, 
        top_percent=filter_top_percent, batch_size=scoring_batch_size
    )
    
    # 第二阶段：在筛选出的候选集中进行最终推荐（Recommendation Stage）
    print(f"🎯 第二阶段（Recommendation Stage）：在 {len(filtered_candidates)} 个候选制品中进行比较推理，生成最终 top-{top_k} 推荐...")
    return _get_llm_recommendations_single_stage(intent, filtered_candidates, ecosystem,
                                                top_k, client, model_name)


def _get_llm_recommendations_single_stage(intent: str, candidates: List[Dict], ecosystem: str,
                                          top_k: int, client: OpenAI, model_name: str) -> List[str]:
    """
    单阶段推荐：直接在候选集中进行推荐
    通过比较推理（comparative reasoning）生成最终的top-k推荐
    """
    candidate_names, candidate_text = build_candidate_text(candidates, max_candidates=None, ecosystem=ecosystem)
    prompt = build_llm_prompt(intent, candidate_text, ecosystem, top_k)
    
    try:
        content = _call_llm_api(client, model_name, prompt)
        recommended_names = []
        
        # 按行解析，提取制品名称
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            # 移除可能的编号前缀（如 "1. ", "- ", "* " 等）
            line = line.lstrip('0123456789.-*()[] ').strip()
            if line and line in candidate_names:
                recommended_names.append(line)
        
        # 如果解析出的推荐数量不足，用剩余的候选制品填充
        if len(recommended_names) < top_k:
            remaining = [name for name in candidate_names if name not in recommended_names]
            recommended_names.extend(remaining[:top_k - len(recommended_names)])
        
        # 确保返回top_k个结果
        return recommended_names[:top_k]
        
    except Exception as e:
        print(f"⚠️ LLM调用出错: {e}")
        # 如果出错，返回前top_k个候选制品作为fallback
        return candidate_names[:top_k]


def evaluate_recommendations(top_names: List[List[str]], candidate_names: List[str],
                             intents: List[str], gold_labels: List[str],
                             p_ks: List[int], dcg_ks: List[int]):
    """计算 P@K 和 DCG@K"""
    metrics = {f"P@{k}": 0.0 for k in p_ks}
    dcg_metrics = {f"DCG@{k}": 0.0 for k in dcg_ks}
    recommendations = []

    for qi, ranked_names in enumerate(top_names):
        rec_entry = {
            "intent": intents[qi],
            "gold": gold_labels[qi],
            "ranking": ranked_names[: max(max(p_ks), max(dcg_ks))],
        }
        recommendations.append(rec_entry)

        for k in p_ks:
            metrics[f"P@{k}"] += precision_at_k(ranked_names, gold_labels[qi], k)
        for k in dcg_ks:
            dcg_metrics[f"DCG@{k}"] += dcg_at_k(ranked_names, gold_labels[qi], k)

    num_q = len(intents)
    for k in p_ks:
        metrics[f"P@{k}"] /= num_q
    for k in dcg_ks:
        dcg_metrics[f"DCG@{k}"] /= num_q

    return metrics, dcg_metrics, recommendations


# ========== 主入口 ==========

def run_llm_recommendation(data_dir: str, ecosystem: str, output_dir: str,
                           model_name: str, top_k: int, p_ks: List[int], 
                           dcg_ks: List[int], max_candidates: int = None,
                           api_key: str = None, base_url: str = None,
                           use_two_stage: bool = True, filter_top_percent: float = 0.1,
                           scoring_batch_size: int = 20):
    """
    使用LLM进行意图需求制品推荐
    
    Args:
        data_dir: 数据目录路径
        ecosystem: 生态系统名称 (hf/js/linux)
        output_dir: 输出目录
        model_name: LLM模型名称
        top_k: 返回的推荐数量
        p_ks: P@K的k值列表
        dcg_ks: DCG@K的k值列表
        max_candidates: 最大候选制品数量（已废弃，保留以兼容）
        api_key: OpenAI API密钥
        base_url: OpenAI API基础URL
        use_two_stage: 是否使用两阶段策略（先打分筛选，再推荐）
        filter_top_percent: 第一阶段筛选出的候选百分比（默认0.1，即top 10%）
        scoring_batch_size: 批量打分的大小（默认20，即每次批量打分20个制品）
    """
    if OpenAI is None:
        raise ImportError("请先安装 openai：pip install openai")
    
    start_time = time.time()
    print(f"⏳ Starting LLM recommendation for [{ecosystem}] using [{model_name}]...")
    
    # 根据模型名称自动创建对应的客户端
    client, actual_model_name = create_llm_client(model_name, api_key, base_url)
    config = get_model_config(model_name, api_key, base_url)
    api_info = config.get("base_url", "default")
    print(f"📡 使用 API: {api_info}")
    
    # 加载数据
    data_path = os.path.join(data_dir, ecosystem)
    dataset = load_json(os.path.join(data_path, "dataset.json"))
    candidates = load_json(os.path.join(data_path, "candidate_artifacts.json"))
    
    intents = [row["intent"].strip() for row in dataset]
    gold_labels = [row["artifact"].strip() for row in dataset]
    
    # 获取推荐结果
    print(f"🚀 开始为 {len(intents)} 个意图生成推荐...")
    if use_two_stage:
        print(f"📋 使用两阶段策略：")
        print(f"   - 第一阶段（Scoring）：对每个制品打分（0-100），产生语义相关性分布")
        print(f"   - 第二阶段（Recommendation）：选择top {filter_top_percent*100:.1f}%候选，通过比较推理生成最终推荐")
    else:
        print(f"📋 使用单阶段策略：直接在全部候选制品中推荐")
    
    all_rankings = []
    query_times: List[float] = []
    intents = intents[:10]
    for idx, intent in enumerate(intents, 1):
        print(f"\n{'='*60}")
        print(f"处理意图 {idx}/{len(intents)}")
        print(f"{'='*60}")
        
        q_start = time.perf_counter()
        recommended_names = get_llm_recommendations(
            intent, candidates, ecosystem, top_k, client, actual_model_name,
            max_candidates=None, use_two_stage=use_two_stage, filter_top_percent=filter_top_percent,
            scoring_batch_size=scoring_batch_size
        )
        all_rankings.append(recommended_names)
        query_times.append(time.perf_counter() - q_start)
        
        # 添加小延迟以避免API限流
        time.sleep(0.2)
    
    # 评估结果
    candidate_names = [c["name"] for c in candidates if "name" in c]
    metrics, dcg_metrics, recommendations = evaluate_recommendations(
        all_rankings, candidate_names, intents, gold_labels, p_ks, dcg_ks
    )
    
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    query_time_mean = round(mean(query_times), 4) if query_times else 0.0
    query_time_std = round(pstdev(query_times), 4) if len(query_times) > 1 else 0.0
    query_time_min = round(min(query_times), 4) if query_times else 0.0
    query_time_max = round(max(query_times), 4) if query_times else 0.0
    
    # 保存结果
    output_dir = os.path.join(output_dir, "LLM")
    os.makedirs(output_dir, exist_ok=True)
    model_dir_name = model_name.replace("/", "_").replace("-", "_")
    output_dir = os.path.join(output_dir, model_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_path = os.path.join(output_dir, f"{ecosystem}-metrics.json")
    rec_path = os.path.join(output_dir, f"{ecosystem}-recommendations.json")
    runtime_summary_path = os.path.join(output_dir, f"{ecosystem}-runtime-summary.txt")
    
    save_json({
        "precision": metrics,
        "dcg": dcg_metrics,
        "runtime_seconds": elapsed_time,
        "query_time_stats": {
            "mean": query_time_mean,
            "std": query_time_std,
            "min": query_time_min,
            "max": query_time_max
        }
    }, metrics_path)
    
    save_json(recommendations, rec_path)
    
    with open(runtime_summary_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {ecosystem:<8} | "
                f"Total: {elapsed_time:>8.2f}s | "
                f"Mean: {query_time_mean:>8.4f}s | "
                f"Std: {query_time_std:>8.4f}s | "
                f"Min: {query_time_min:>8.4f}s | "
                f"Max: {query_time_max:>8.4f}s\n")
    
    print("✅ Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    for k, v in dcg_metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"⏱️ Runtime: {elapsed_time:.2f} seconds")
    print("Query time stats (seconds):")
    print(f"  Mean: {query_time_mean:.4f}")
    print(f"  Std : {query_time_std:.4f}")
    print(f"  Min : {query_time_min:.4f}")
    print(f"  Max : {query_time_max:.4f}")
    print(f"📝 Saved to: {metrics_path}\n")


def get_parser():
    parser = argparse.ArgumentParser(description="使用大语言模型进行意图需求制品推荐")
    parser.add_argument("--data_dir", type=str, default="IntentRecBench/data",
                        help="数据目录路径")
    parser.add_argument("--ecosystem", type=str, default="hf", 
                        choices=["hf", "js", "linux"],
                        help="生态系统名称")
    parser.add_argument("--output_dir", type=str, default="output/baselines",
                        help="输出目录")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="LLM模型名称 (支持: gpt-*, qwen/*, deepseek/*, llama/*)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="返回的推荐数量")
    parser.add_argument("--p_k", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="P@K的k值列表")
    parser.add_argument("--dcg_k", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="DCG@K的k值列表")
    parser.add_argument("--max_candidates", type=int, default=None,
                        help="最大候选制品数量（已废弃，保留以兼容）")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API密钥（如果不设置，将使用环境变量OPENAI_API_KEY）")
    parser.add_argument("--base_url", type=str, default=None,
                        help="OpenAI API基础URL（如果不设置，将使用环境变量OPENAI_BASE_URL）")
    parser.add_argument("--use_two_stage", action="store_true", default=True,
                        help="使用两阶段策略：先对每个制品打分筛选，再进行最终推荐（默认启用）")
    parser.add_argument("--no_two_stage", dest="use_two_stage", action="store_false",
                        help="禁用两阶段策略，直接使用单阶段推荐")
    parser.add_argument("--filter_top_percent", type=float, default=0.1,
                        help="第一阶段筛选出的候选制品百分比（默认0.1，即top 10%）")
    parser.add_argument("--scoring_batch_size", type=int, default=20,
                        help="批量打分的大小（默认20，即每次批量打分20个制品，可显著减少API调用次数）")
    return parser


def main():
    args = get_parser().parse_args()
    
    # 从环境变量或参数中获取API配置

    run_llm_recommendation(
        data_dir=args.data_dir,
        ecosystem=args.ecosystem,
        output_dir=args.output_dir,
        model_name=args.model_name,
        top_k=args.top_k,
        p_ks=args.p_k,
        dcg_ks=args.dcg_k,
        max_candidates=args.max_candidates,
        api_key=args.api_key,
        base_url=args.base_url,
        use_two_stage=args.use_two_stage,
        filter_top_percent=args.filter_top_percent,
        scoring_batch_size=args.scoring_batch_size
    )


if __name__ == "__main__":
    main()
