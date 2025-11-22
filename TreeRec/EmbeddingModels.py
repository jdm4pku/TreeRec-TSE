import logging
from abc import ABC, abstractmethod
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# ========== 嵌入模型加载 ==========
def sentence_transformers_load_embedding_model(model_name: str):
    # === SentenceTransformer 模型加载 ===
    if SentenceTransformer is None:
        raise ImportError("请先安装 sentence-transformers：pip install sentence-transformers")

    try:
        # ✅ 优先尝试从 ModelScope 加载
        from modelscope import snapshot_download
        print(f"📦 Downloading model [{model_name}] from ModelScope ...")
        model_dir = snapshot_download(model_name)
        model = SentenceTransformer(model_dir)
        print("✅ Model loaded from ModelScope.")
    except Exception as e:
        print(f"⚠️ ModelScope 加载失败 ({e})，改用 Hugging Face 方式。")
        model = SentenceTransformer(model_name)

    return model

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI(
            base_url = "https://openrouter.ai/api/v1",
            api_key = ""
        )
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )

class QwenEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="Qwen/Qwen3-Embedding-8B"):
        self.client = OpenAI(
            base_url = "https://api.siliconflow.cn/v1",
            api_key = ""
        )
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )

class SentenceTransformersEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        # self.model = model
        self.model = sentence_transformers_load_embedding_model(model)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        return self.model.encode(text, normalize_embeddings=True)