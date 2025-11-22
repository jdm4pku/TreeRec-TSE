import logging
import pickle
import json
import re

from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import BaseEmbeddingModel
from .RerankModels import BaseRerankModel, GPT3TurboRerankModel, QwenRerankModel, DeepSeekRerankModel, LlamaRerankModel, GPT4oRerankModel
from .SummarizationModels import BaseSummarizationModel, GPT4oSummarizationModel, QwenSummarizationModel, DeepSeekSummarizationModel, LlamaSummarizationModel
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

# Define a dictionary to map supported tree builders to their respective configs
supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def parse_rerank_result(answer_str: str) -> dict:
    """
    解析 rerank 模型返回的结果，处理各种格式问题。
    
    Args:
        answer_str: rerank 模型返回的字符串
        
    Returns:
        解析后的 JSON 字典
        
    Raises:
        json.JSONDecodeError: 如果无法解析为有效的 JSON
    """
    if not answer_str or not isinstance(answer_str, str):
        raise ValueError("Answer string is empty or not a string")
    
    # 1. 尝试直接解析 JSON
    try:
        return json.loads(answer_str.strip())
    except json.JSONDecodeError:
        pass
    
    # 2. 尝试从 markdown 代码块中提取 JSON
    # 匹配 ```json ... ``` 或 ``` ... ```
    # 先找到代码块，然后提取其中的 JSON
    code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
    code_block_match = re.search(code_block_pattern, answer_str, re.DOTALL)
    if code_block_match:
        code_content = code_block_match.group(1).strip()
        # 在代码块内容中查找 JSON 对象
        json_match = re.search(r'\{.*\}', code_content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0).strip())
            except json.JSONDecodeError:
                pass
    
    # 3. 尝试提取第一个 JSON 对象（使用大括号匹配）
    # 找到第一个 { 和最后一个 }
    start_idx = answer_str.find('{')
    if start_idx != -1:
        # 从第一个 { 开始，找到匹配的 }
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(answer_str)):
            if answer_str[i] == '{':
                brace_count += 1
            elif answer_str[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        if end_idx > start_idx:
            try:
                json_str = answer_str[start_idx:end_idx]
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                pass
    
    # 4. 尝试修复常见的 JSON 格式问题
    # 移除可能的 markdown 格式标记
    cleaned = re.sub(r'```[a-z]*\s*', '', answer_str)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()
    
    # 尝试提取 JSON 部分（去除前后可能的文本）
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0).strip())
        except json.JSONDecodeError:
            pass
    
    # 如果所有方法都失败，抛出异常
    raise json.JSONDecodeError(
        f"Unable to parse JSON from rerank result. First 200 chars: {answer_str[:200]}",
        answer_str, 0
    )


class RetrievalAugmentationConfig:
    def __init__(
        self,
        tree_builder_config=None,
        tree_retriever_config=None,  # Change from default instantiation
        rerank_model=None,
        embedding_model=None,
        summarization_model=None,
        tree_builder_type="cluster",
        use_rerank=False,
        # New parameters for TreeRetrieverConfig and TreeBuilderConfig
        # TreeRetrieverConfig arguments
        tr_tokenizer=None,
        tr_threshold=0.5,
        tr_top_k=5,
        tr_selection_mode="top_k",
        tr_context_embedding_model="OpenAI",
        tr_embedding_model=None,
        tr_num_layers=None,
        tr_start_layer=None,
        # TreeBuilderConfig arguments
        tb_tokenizer=None,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_summarization_model=None,
        tb_embedding_model=None,
        tb_cluster_embedding_model="OpenAI",
        **kwargs,
    ):
        if tr_context_embedding_model is None:
            tr_context_embedding_model = embedding_model
        if tr_embedding_model is None:
            tr_embedding_model = embedding_model
        if tb_summarization_model is None:
            tb_summarization_model = summarization_model
        if tb_embedding_model is None:
            tb_embedding_model = embedding_model
        if tb_cluster_embedding_model is None:
            tb_cluster_embedding_model = embedding_model
        
        # Set TreeBuilderConfig
        tree_builder_class, tree_builder_config_class = supported_tree_builders[
            tree_builder_type
        ]
        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                embedding_model=tb_embedding_model,
                cluster_embedding_model=tb_cluster_embedding_model,
            )

        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'"
            )

        # Set TreeRetrieverConfig
        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model,
                embedding_model=tr_embedding_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer,
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError(
                "tree_retriever_config must be an instance of TreeRetrieverConfig"
            )

        # Assign the created configurations to the instance
        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.use_rerank = use_rerank
        if rerank_model and self.use_rerank:
            if rerank_model.startswith("Qwen"):
                self.rerank_model = QwenRerankModel(rerank_model)
            elif summarization_model.startswith("Pro/deepseek"):
                self.rerank_model = DeepSeekRerankModel(rerank_model)
            elif summarization_model.startswith("meta-llama"):
                self.rerank_model = LlamaRerankModel(rerank_model)
            elif summarization_model.startswith("gpt"):
                self.rerank_model = GPT4oRerankModel(rerank_model)
            else:
                raise ValueError(f"Unsupported rerank model: {rerank_model}")
        else:
            self.rerank_model = None
        self.tree_builder_type = tree_builder_type

    def log_config(self):
        config_summary = """
        RetrievalAugmentationConfig:
            {tree_builder_config}
            
            {tree_retriever_config}
            
            Rerank Model: {rerank_model}
            Use Rerank: {use_rerank}
            Tree Builder Type: {tree_builder_type}
        """.format(
            tree_builder_config=self.tree_builder_config.log_config(),
            tree_retriever_config=self.tree_retriever_config.log_config(),
            rerank_model=self.rerank_model,
            use_rerank=self.use_rerank,
            tree_builder_type=self.tree_builder_type,
        )
        return config_summary       


class RetrievalAugmentation:
    """
    A Retrieval Augmentation class that combines the TreeBuilder and TreeRetriever classes.
    Enables adding documents to the tree, retrieving information, and answering questions.
    """

    def __init__(self, config=None, tree=None):
        """
        Initializes a RetrievalAugmentation instance with the specified configuration.
        Args:
            config (RetrievalAugmentationConfig): The configuration for the RetrievalAugmentation instance.
            tree: The tree instance or the path to a pickled tree file.
        """
        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError(
                "config must be an instance of RetrievalAugmentationConfig"
            )

        # Check if tree is a string (indicating a path to a pickled tree)
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.rerank_model = config.rerank_model
        self.use_rerank = config.use_rerank

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info(
            f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}"
        )

    def add_artifacts(self, artifacts):
        """
        Adds documents to the tree and creates a TreeRetriever instance.

        Args:
            artifacts (list): [name:xx, description:xx] 
        """
        if self.tree is not None:
            logging.warning("Overwriting existing tree. Did you mean to call 'add_to_existing' instead?")
            return

        self.tree = self.tree_builder.build_from_artifact(artifacts)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
    
    def retrieve(
        self,
        question,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = False,
        return_layer_information: bool = True,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            question (str): The question to answer.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            use_all_information (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The context from which the answer can be found.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        if self.retriever is None:
            raise ValueError(
                "The TreeRetriever instance has not been initialized. Call 'add_documents' first."
            )

        return self.retriever.retrieve(
            question,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            collapse_tree,
            return_layer_information,
        )

    def artifact_recommendation(
        self,
        intent,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 3500,
        collapse_tree: bool = False,
        return_layer_information: bool = False,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            question (str): The question to answer.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            use_all_information (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The answer to the question.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        # if return_layer_information:
        top_k_artifacts, top_k_context = self.retrieve(
            intent, start_layer, num_layers, top_k, max_tokens, collapse_tree, return_layer_information
        )

        # 根据 use_rerank 参数决定是否使用 rerank
        if self.rerank_model is not None and self.use_rerank:
            # 使用 rerank model
            answer = self.rerank_model.rerank(top_k_context, intent)
            # parse the answer to get the top k artifacts, answer now is a str
            try:
                answer_dict = parse_rerank_result(answer)
                reranked_top_k_artifacts = answer_dict.get("reranked_artifacts", [])
                
                # 验证结果是否为列表且包含字符串
                if not isinstance(reranked_top_k_artifacts, list):
                    raise ValueError(f"reranked_artifacts is not a list: {type(reranked_top_k_artifacts)}")
                
                # 如果 rerank 返回的结果数量不足，用原始结果补充
                if len(reranked_top_k_artifacts) < top_k:
                    original_names = [artifact.name for artifact in top_k_artifacts]
                    for name in original_names:
                        if name not in reranked_top_k_artifacts:
                            reranked_top_k_artifacts.append(name)
                            if len(reranked_top_k_artifacts) >= top_k:
                                break
                return reranked_top_k_artifacts[:top_k]
            except (json.JSONDecodeError, ValueError, AttributeError, KeyError) as e:
                logging.warning(
                    f"Failed to parse rerank result for intent '{intent[:50]}...': {e}. "
                    f"Answer preview: {str(answer)[:200] if answer else 'None'}. "
                    f"Using original ranking."
                )
                # 如果解析失败，回退到原始结果
                top_k_artifacts_name = []
                for artifact in top_k_artifacts:
                    top_k_artifacts_name.append(artifact.name)
                return top_k_artifacts_name
        else:
            # 不使用 rerank model，直接返回原始结果
            top_k_artifacts_name = []
            for artifact in top_k_artifacts:
                top_k_artifacts_name.append(artifact.name)
            return top_k_artifacts_name

    def save(self, path):
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {path}")