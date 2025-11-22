import copy
import logging
import os
from abc import abstractclassmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .EmbeddingModels import (BaseEmbeddingModel, OpenAIEmbeddingModel,
                              QwenEmbeddingModel, SentenceTransformersEmbeddingModel)
from .SummarizationModels import (BaseSummarizationModel,
                                  GPT4oSummarizationModel,
                                  QwenSummarizationModel,
                                  DeepSeekSummarizationModel,
                                  LlamaSummarizationModel)
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeBuilderConfig:
    def __init__(
        self,
        tokenizer=None,
        max_tokens=None,
        num_layers=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        summarization_length=None,
        summarization_model=None,
        embedding_model=None,
        cluster_embedding_model=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        if max_tokens is None:
            max_tokens = 100
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")
        self.max_tokens = max_tokens

        if num_layers is None:
            num_layers = 5
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")
        self.num_layers = num_layers

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a number between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if selection_mode not in ["top_k", "threshold"]:
            raise ValueError("selection_mode must be either 'top_k' or 'threshold'")
        self.selection_mode = selection_mode

        if summarization_length is None:
            summarization_length = 100
        self.summarization_length = summarization_length

        if summarization_model.startswith("Qwen"):
            self.summarization_model = QwenSummarizationModel(summarization_model)
        elif summarization_model.startswith("Pro/deepseek"):
            self.summarization_model = DeepSeekSummarizationModel(summarization_model)
        elif summarization_model.startswith("meta-llama"):
            self.summarization_model = LlamaSummarizationModel(summarization_model)
        elif summarization_model.startswith("gpt"):
            self.summarization_model = GPT4oSummarizationModel(summarization_model)
        else:
            raise ValueError(f"Unsupported summarization model: {summarization_model}")

        if embedding_model.startswith("Qwen"):
            self.embedding_model = QwenEmbeddingModel()
        elif embedding_model.startswith("sentence-transformers"):
            self.embedding_model = SentenceTransformersEmbeddingModel(embedding_model)
        elif embedding_model.startswith("text-embedding-ada-002"):
            self.embedding_model = OpenAIEmbeddingModel(embedding_model)
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model}")
        
        if cluster_embedding_model.startswith("Qwen"):
            self.cluster_embedding_model = QwenEmbeddingModel(cluster_embedding_model)
        elif cluster_embedding_model.startswith("sentence-transformers"):
            self.cluster_embedding_model = SentenceTransformersEmbeddingModel(cluster_embedding_model)
        elif cluster_embedding_model.startswith("text-embedding-ada-002"):
            self.cluster_embedding_model = OpenAIEmbeddingModel(cluster_embedding_model)
        else:
            raise ValueError(f"Unsupported cluster embedding model: {cluster_embedding_model}")

    def log_config(self):
        config_log = """
        TreeBuilderConfig:
            Tokenizer: {tokenizer}
            Max Tokens: {max_tokens}
            Num Layers: {num_layers}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Summarization Length: {summarization_length}
            Summarization Model: {summarization_model}
            Embedding Models: {embedding_model}
        """.format(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            num_layers=self.num_layers,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            summarization_length=self.summarization_length,
            summarization_model=self.summarization_model,
            embedding_model=self.embedding_model,
        )
        return config_log


class TreeBuilder:
    """
    The TreeBuilder class is responsible for building a hierarchical text abstraction
    structure, known as a "tree," using summarization models and
    embedding models.
    """

    def __init__(self, config) -> None:
        """Initializes the tokenizer, maximum tokens, number of layers, top-k value, threshold, and selection mode."""

        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.num_layers = config.num_layers
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_model = config.embedding_model
        self.cluster_embedding_model = config.cluster_embedding_model

        logging.info(
            f"Successfully initialized TreeBuilder with Config {config.log_config()}"
        )

    def create_node(
        self,
        index: int,
        name: str,
        desc: str,
        children_indices: Optional[Set[int]] = None,
    ) -> Tuple[int, Node]:
        """Creates a new node with name and description, and optional children.

        Embeddings are created on the concatenation of name and description.
        """
        if children_indices is None:
            children_indices = set()

        combined_text = name if desc == "" else (name + ": " + desc) if name != "" else desc

        # Create embedding using the single embedding model
        embedding = self.embedding_model.create_embedding(combined_text)
        # Store as dictionary with model name as key for compatibility
        if isinstance(self.embedding_model, QwenEmbeddingModel):
            embedding_model_name = "Qwen"
        else:
            embedding_model_name = "OpenAI"
        embeddings = {
            embedding_model_name: embedding
        }
        return (index, Node(name, desc, index, children_indices, embedding))

    def create_embedding(self, text) -> List[float]:
        """
        Generates embeddings for the given text using the specified embedding model.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            List[float]: The generated embeddings.
        """
        return self.cluster_embedding_model.create_embedding(text)

    def summarize(self, context, max_tokens=150) -> str:
        """
        Generates a summary of the input context using the specified summarization model.

        Args:
            context (str, optional): The context to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.o

        Returns:
            str: The generated summary.
        """
        return self.summarization_model.summarize(context, max_tokens)

    def get_relevant_nodes(self, current_node, list_nodes) -> List[Node]:
        """
        Retrieves the top-k most relevant nodes to the current node from the list of nodes
        based on cosine distance in the embedding space.

        Args:
            current_node (Node): The current node.
            list_nodes (List[Node]): The list of nodes.

        Returns:
            List[Node]: The top-k most relevant nodes.
        """
        embeddings = get_embeddings(list_nodes, self.cluster_embedding_model_name)
        distances = distances_from_embeddings(
            current_node.embeddings[self.cluster_embedding_model], embeddings
        )
        indices = indices_of_nearest_neighbors_from_distances(distances)

        if self.selection_mode == "threshold":
            best_indices = [
                index for index in indices if distances[index] > self.threshold
            ]

        elif self.selection_mode == "top_k":
            best_indices = indices[: self.top_k]

        nodes_to_add = [list_nodes[idx] for idx in best_indices]

        return nodes_to_add

    def multithreaded_create_leaf_nodes(self, leaves: List[Tuple[str, str]]) -> Dict[int, Node]:
        """Creates leaf nodes using multithreading from the given list of (name, desc).

        Args:
            chunks (List[str]): A list of text chunks to be turned into leaf nodes.

        Returns:
            Dict[int, Node]: A dictionary mapping node indices to the corresponding leaf nodes.
        """
        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, name, desc): (index, name, desc)
                for index, (name, desc) in enumerate(leaves)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes

    def build_from_text(self, text: str, use_multithreading: bool = True) -> Tree:
        """Builds a golden tree from the input text, optionally using multithreading.

        Args:
            text (str): The input text.
            use_multithreading (bool, optional): Whether to use multithreading when creating leaf nodes.
                Default: True.

        Returns:
            Tree: The golden tree structure.
        """
        chunks = split_text(text, self.tokenizer, self.max_tokens)

        logging.info("Creating Leaf Nodes")

        if use_multithreading:
            leaves = [("", chunk) for chunk in chunks]
            leaf_nodes = self.multithreaded_create_leaf_nodes(leaves)
        else:
            leaf_nodes = {}
            for index, chunk in enumerate(chunks):
                __, node = self.create_node(index, "", chunk)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}

        logging.info(f"Created {len(leaf_nodes)} Leaf Embeddings")

        logging.info("Building All Nodes")

        all_nodes = copy.deepcopy(leaf_nodes)

        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)

        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)

        return tree

    def build_from_artifact(self, artifacts: List[Dict[str, str]], use_multithreading: bool = False) -> Tree:
        """Builds a tree from a list of artifacts with fields name and description.

        artifacts: List of dicts with keys: name, description (desc also accepted).
        """
        leaves: List[Tuple[str, str]] = []
        for art in artifacts:
            name = art.get("name", "")
            desc = art.get("description", art.get("desc", ""))
            leaves.append((name, desc))

        logging.info("Creating Leaf Nodes from artifacts")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(leaves)
        else:
            leaf_nodes = {}
            for index, (name, desc) in enumerate(leaves):
                __, node = self.create_node(index, name, desc)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}

        logging.info(f"Created {len(leaf_nodes)} Leaf Embeddings")

        logging.info("Building All Nodes")

        all_nodes = copy.deepcopy(leaf_nodes)

        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)

        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)

        return tree

    @abstractclassmethod
    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        """
        Constructs the hierarchical tree structure layer by layer by iteratively summarizing groups
        of relevant nodes and updating the current_level_nodes and all_tree_nodes dictionaries at each step.

        Args:
            current_level_nodes (Dict[int, Node]): The current set of nodes.
            all_tree_nodes (Dict[int, Node]): The dictionary of all nodes.
            use_multithreading (bool): Whether to use multithreading to speed up the process.

        Returns:
            Dict[int, Node]: The final set of root nodes.
        """
        pass

