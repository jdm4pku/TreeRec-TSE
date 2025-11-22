import os
import pickle
import json
from typing import Dict, List, Set
from scipy import spatial
from TreeRec.tree_structures import Tree, Node


def find_parent_nodes(tree: Tree, leaf_index: int) -> Set[int]:
    """
    找到包含指定叶子节点的所有父节点
    
    Args:
        tree: 树对象
        leaf_index: 叶子节点索引
        
    Returns:
        包含该叶子节点的父节点索引集合
    """
    parent_indices = set()
    for node_index, node in tree.all_nodes.items():
        if leaf_index in node.children:
            parent_indices.add(node_index)
    return parent_indices


def get_leaf_clusters(tree: Tree) -> Dict[int, Set[int]]:
    """
    根据直接父节点将叶子节点分组为簇
    
    Args:
        tree: 树对象
        
    Returns:
        字典，键为父节点索引（或-1表示根节点），值为该簇的叶子节点索引集合
    """
    leaf_indices = set(tree.leaf_nodes.keys())
    clusters = {}  # parent_index -> set of leaf indices
    
    # 找到每个叶子节点的直接父节点
    for leaf_idx in leaf_indices:
        parents = find_parent_nodes(tree, leaf_idx)
        
        if not parents:
            # 如果没有父节点，说明是根节点（不太可能，但处理一下）
            cluster_key = -1
        else:
            # 使用最小的父节点索引作为簇标识（或者可以按层选择最近的父节点）
            # 这里选择第一个父节点作为簇标识
            cluster_key = min(parents)
        
        if cluster_key not in clusters:
            clusters[cluster_key] = set()
        clusters[cluster_key].add(leaf_idx)
    
    return clusters


def calculate_silhouette_coefficient(tree: Tree) -> float:
    """
    计算树的轮廓系数
    
    对于每个叶子节点：
    - a(i): 与同一簇内其他叶子节点的平均距离
    - b(i): 与最近的其他簇的平均距离
    - s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    整体轮廓系数 = 所有叶子节点的 s(i) 的平均值
    
    Args:
        tree: 树对象
        
    Returns:
        轮廓系数值
    """
    leaf_indices = list(tree.leaf_nodes.keys())
    
    if len(leaf_indices) < 2:
        return 0.0  # 至少需要2个叶子节点才能计算轮廓系数
    
    # 获取叶子节点的簇分配
    clusters = get_leaf_clusters(tree)
    
    # 如果只有一个簇，无法计算轮廓系数
    if len(clusters) < 2:
        return 0.0
    
    # 为每个叶子节点找到其所属的簇
    leaf_to_cluster = {}
    for cluster_key, leaf_set in clusters.items():
        for leaf_idx in leaf_set:
            leaf_to_cluster[leaf_idx] = cluster_key
    
    # 计算每个叶子节点的轮廓系数
    silhouette_scores = []
    
    for leaf_idx in leaf_indices:
        leaf_node = tree.leaf_nodes[leaf_idx]
        leaf_embedding = leaf_node.embedding
        
        if leaf_embedding is None:
            continue
        
        # 找到该叶子节点所属的簇
        cluster_key = leaf_to_cluster[leaf_idx]
        same_cluster_leaves = clusters[cluster_key]
        
        # 计算 a(i): 与同一簇内其他叶子节点的平均距离
        same_cluster_distances = []
        for other_leaf_idx in same_cluster_leaves:
            if other_leaf_idx != leaf_idx:
                other_leaf_node = tree.leaf_nodes[other_leaf_idx]
                if other_leaf_node.embedding is not None:
                    distance = spatial.distance.cosine(leaf_embedding, other_leaf_node.embedding)
                    same_cluster_distances.append(distance)
        
        a_i = sum(same_cluster_distances) / len(same_cluster_distances) if same_cluster_distances else 0.0
        
        # 计算 b(i): 与最近的其他簇的平均距离
        other_cluster_avg_distances = []
        for other_cluster_key, other_cluster_leaves in clusters.items():
            if other_cluster_key != cluster_key:
                other_cluster_distances = []
                for other_leaf_idx in other_cluster_leaves:
                    other_leaf_node = tree.leaf_nodes[other_leaf_idx]
                    if other_leaf_node.embedding is not None:
                        distance = spatial.distance.cosine(leaf_embedding, other_leaf_node.embedding)
                        other_cluster_distances.append(distance)
                
                if other_cluster_distances:
                    avg_distance = sum(other_cluster_distances) / len(other_cluster_distances)
                    other_cluster_avg_distances.append(avg_distance)
        
        b_i = min(other_cluster_avg_distances) if other_cluster_avg_distances else 0.0
        
        # 计算 s(i)
        if max(a_i, b_i) == 0:
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)
        
        silhouette_scores.append(s_i)
    
    # 返回平均轮廓系数
    if not silhouette_scores:
        return 0.0
    return sum(silhouette_scores) / len(silhouette_scores)


def analyze_tree(tree_path: str) -> Dict:
    """
    分析树的结构信息
    
    Args:
        tree_path: 树文件的路径
        
    Returns:
        包含统计信息的字典
    """
    # 加载树
    with open(tree_path, "rb") as f:
        tree: Tree = pickle.load(f)
    
    # 获取叶子节点索引集合
    leaf_indices = set(tree.leaf_nodes.keys())
    
    # 统计叶子节点和非叶子节点的 description 长度
    leaf_desc_lengths = []
    non_leaf_desc_lengths = []
    
    for index, node in tree.all_nodes.items():
        desc_length = len(node.desc) if node.desc else 0
        if index in leaf_indices:
            leaf_desc_lengths.append(desc_length)
        else:
            non_leaf_desc_lengths.append(desc_length)
    
    # 计算平均值
    avg_leaf_desc_length = sum(leaf_desc_lengths) / len(leaf_desc_lengths) if leaf_desc_lengths else 0
    avg_non_leaf_desc_length = sum(non_leaf_desc_lengths) / len(non_leaf_desc_lengths) if non_leaf_desc_lengths else 0
    
    # 计算轮廓系数
    silhouette_coefficient = calculate_silhouette_coefficient(tree)
    
    # 统计信息
    stats = {
        "num_nodes": len(tree.all_nodes),
        "num_leaf_nodes": len(tree.leaf_nodes),
        "num_non_leaf_nodes": len(tree.all_nodes) - len(tree.leaf_nodes),
        "num_layers": tree.num_layers,
        "avg_leaf_desc_length": round(avg_leaf_desc_length, 2),
        "avg_non_leaf_desc_length": round(avg_non_leaf_desc_length, 2),
        "silhouette_coefficient": round(silhouette_coefficient, 4),
        "min_leaf_desc_length": min(leaf_desc_lengths) if leaf_desc_lengths else 0,
        "max_leaf_desc_length": max(leaf_desc_lengths) if leaf_desc_lengths else 0,
        "min_non_leaf_desc_length": min(non_leaf_desc_lengths) if non_leaf_desc_lengths else 0,
        "max_non_leaf_desc_length": max(non_leaf_desc_lengths) if non_leaf_desc_lengths else 0,
    }
    
    return stats


def main():
    """统计 GPT_4 为每个生态构建的树信息"""
    # 定义生态和树文件路径
    base_dir = "output/TreeRec/gpt4o"
    ecosystems = ["js", "linux"]
    
    all_stats = {}
    
    print("=" * 60)
    print("统计 GPT_4 构建的树信息")
    print("=" * 60)
    
    for ecosystem in ecosystems:
        tree_path = os.path.join(base_dir, f"{ecosystem}-tree.pkl")
        
        if not os.path.exists(tree_path):
            print(f"⚠️  警告: 树文件不存在: {tree_path}")
            continue
        
        print(f"\n📊 分析生态: {ecosystem}")
        print(f"   树文件: {tree_path}")
        
        stats = analyze_tree(tree_path)
        all_stats[ecosystem] = stats
        
        # 打印统计信息
        print(f"   节点数: {stats['num_nodes']}")
        print(f"   - 叶子节点数: {stats['num_leaf_nodes']}")
        print(f"   - 非叶子节点数: {stats['num_non_leaf_nodes']}")
        print(f"   层数: {stats['num_layers']}")
        print(f"   叶子节点 description 平均长度: {stats['avg_leaf_desc_length']}")
        print(f"   非叶子节点 description 平均长度: {stats['avg_non_leaf_desc_length']}")
        print(f"   轮廓系数: {stats['silhouette_coefficient']}")
    
    # 保存统计结果到 JSON 文件
    output_path = os.path.join(base_dir, "tree_statistics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 统计结果已保存到: {output_path}")
    
    # 打印汇总表格
    print("\n" + "=" * 100)
    print("汇总表格")
    print("=" * 100)
    print(f"{'生态':<10} {'节点数':<10} {'层数':<8} {'叶子节点desc平均长度':<20} {'非叶子节点desc平均长度':<25} {'轮廓系数':<12}")
    print("-" * 100)
    for ecosystem in ecosystems:
        if ecosystem in all_stats:
            stats = all_stats[ecosystem]
            print(f"{ecosystem:<10} {stats['num_nodes']:<10} {stats['num_layers']:<8} "
                  f"{stats['avg_leaf_desc_length']:<20} {stats['avg_non_leaf_desc_length']:<25} "
                  f"{stats['silhouette_coefficient']:<12}")


if __name__ == "__main__":
    main()

