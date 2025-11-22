
import logging
import os

from openai import OpenAI


import getpass
from abc import ABC, abstractmethod

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import T5ForConditionalGeneration, T5Tokenizer


class BaseRerankModel(ABC):
    @abstractmethod
    def rerank(self, context, question):
        """
        Reranks the given context using the model.
        """
        pass


class GPT3TurboRerankModel(BaseRerankModel):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def rerank(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.completions.create(
                prompt=f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response.choices[0].text.strip()

        except Exception as e:
            print(e)
            return ""


class GPT3TurboRerankModel(BaseRerankModel):
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_rerank(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in software artifact retrieval and relevance ranking. "
                        "Your task is to reorder retrieved artifacts based on their semantic relevance to the user's intent."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User intent (query): {question}\n\n"
                        f"Top-k retrieved artifacts:\n{context}\n\n"
                        f"Please re-rank the artifacts from most relevant to least relevant "
                        f"according to how well each artifact satisfies the user's intent.\n\n"
                        f"Output the result strictly in **JSON format**, with the following structure:\n\n"
                        f"{{\n"
                        f'  "reranked_artifacts": ["artifact_name_1", "artifact_name_2", "artifact_name_3", ...]\n'
                        f"}}\n\n"
                        f"Do not include explanations, scores, or any additional text outside the JSON object."
                    ),
                },
            ],
            temperature=0,
        )
        
        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def rerank(self, context, question, max_tokens=150, stop_sequence=None):
        try:
            return self._attempt_rerank(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def rerank(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_rerank(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class GPT4oRerankModel(BaseRerankModel):
    def __init__(self, model="gpt-4o-2024-05-13"):
        """
        Initializes the GPT-4o model with the specified model version.
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_rerank(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in software artifact retrieval and relevance ranking. "
                        "Your task is to reorder retrieved artifacts based on their semantic relevance to the user's intent."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User intent (query): {question}\n\n"
                        f"Top-k retrieved artifacts:\n{context}\n\n"
                        f"Please re-rank the artifacts from most relevant to least relevant "
                        f"according to how well each artifact satisfies the user's intent.\n\n"
                        f"Output the result strictly in **JSON format**, with the following structure:\n\n"
                        f"{{\n"
                        f'  "reranked_artifacts": ["artifact_name_1", "artifact_name_2", "artifact_name_3", ...]\n'
                        f"}}\n\n"
                        f"Do not include explanations, scores, or any additional text outside the JSON object."
                    ),
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def rerank(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_rerank(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e

class QwenRerankModel(BaseRerankModel):
    def __init__(self, model="Qwen/Qwen3-14B"):
        """
        Initializes the GPT-4 model with the specified model version.
        """
        self.model = model
        self.client = OpenAI(
            base_url = "https://api.siliconflow.cn/v1",
            api_key = "sk-wbnxvocaaofhilzlgkvhiuhoivdawabyvaavkvblnokomdyz"
        )
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_rerank(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in software artifact retrieval and relevance ranking. "
                        "Your task is to reorder retrieved artifacts based on their semantic relevance to the user's intent."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User intent (query): {question}\n\n"
                        f"Top-k retrieved artifacts:\n{context}\n\n"
                        f"Please re-rank the artifacts from most relevant to least relevant "
                        f"according to how well each artifact satisfies the user's intent.\n\n"
                        f"Output the result strictly in **JSON format**, with the following structure:\n\n"
                        f"{{\n"
                        f'  "reranked_artifacts": ["artifact_name_1", "artifact_name_2", "artifact_name_3", ...]\n'
                        f"}}\n\n"
                        f"Do not include explanations, scores, or any additional text outside the JSON object."
                    ),
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def rerank(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_rerank(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e

class DeepSeekRerankModel(BaseRerankModel):
    def __init__(self, model="Pro/deepseek-ai/DeepSeek-R1"):
        """
        Initializes the DeepSeek model with the specified model version.
        """
        self.client = OpenAI(
            base_url = "https://api.siliconflow.cn/v1",
            api_key = "sk-wbnxvocaaofhilzlgkvhiuhoivdawabyvaavkvblnokomdyz"
        )
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_rerank(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in software artifact retrieval and relevance ranking. "
                        "Your task is to reorder retrieved artifacts based on their semantic relevance to the user's intent."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User intent (query): {question}\n\n"
                        f"Top-k retrieved artifacts:\n{context}\n\n"
                        f"Please re-rank the artifacts from most relevant to least relevant "
                        f"according to how well each artifact satisfies the user's intent.\n\n"
                        f"Output the result strictly in **JSON format**, with the following structure:\n\n"
                        f"{{\n"
                        f'  "reranked_artifacts": ["artifact_name_1", "artifact_name_2", "artifact_name_3", ...]\n'
                        f"}}\n\n"
                        f"Do not include explanations, scores, or any additional text outside the JSON object."
                    ),
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def rerank(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_rerank(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e

class LlamaRerankModel(BaseRerankModel):
    def __init__(self, model="meta-llama/llama-3.1-8b-instruct"):
        """
        Initializes the Llama model with the specified model version.
        """
        self.client = OpenAI(
            base_url = "https://openrouter.ai/api/v1",
            api_key = "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"
        )
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_rerank(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in software artifact retrieval and relevance ranking. "
                        "Your task is to reorder retrieved artifacts based on their semantic relevance to the user's intent."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User intent (query): {question}\n\n"
                        f"Top-k retrieved artifacts:\n{context}\n\n"
                        f"Please re-rank the artifacts from most relevant to least relevant "
                        f"according to how well each artifact satisfies the user's intent.\n\n"
                        f"Output the result strictly in **JSON format**, with the following structure:\n\n"
                        f"{{\n"
                        f'  "reranked_artifacts": ["artifact_name_1", "artifact_name_2", "artifact_name_3", ...]\n'
                        f"}}\n\n"
                        f"Do not include explanations, scores, or any additional text outside the JSON object."
                    ),
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def rerank(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_rerank(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e

# class UnifiedQAModel(BaseRerankModel):
#     def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
#             self.device
#         )
#         self.tokenizer = T5Tokenizer.from_pretrained(model_name)

#     def run_model(self, input_string, **generator_args):
#         input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
#             self.device
#         )
#         res = self.model.generate(input_ids, **generator_args)
#         return self.tokenizer.batch_decode(res, skip_special_tokens=True)

#     def rerank(self, context, question):
#         input_string = question + " \\n " + context
#         output = self.run_model(input_string)
#         return output[0]
