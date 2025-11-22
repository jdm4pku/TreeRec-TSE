from typing import Dict, List, Set

MAX_DESC_LEN = 1000


class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(self, name: str, desc: str, index: int, children: Set[int], embedding) -> None:
        self.name = name
        # 限制描述长度在 3000 字符以内（可根据需要调整）
        if len(desc) > MAX_DESC_LEN:
            desc = desc[:MAX_DESC_LEN] + " ... (truncated)"
        self.desc = desc
        self.index = index
        self.children = children
        self.embedding = embedding


class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes