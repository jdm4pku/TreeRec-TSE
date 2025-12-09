
import logging
import os

from openai import OpenAI


import getpass
from abc import ABC, abstractmethod

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .utils import load_prompt


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
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("请设置环境变量 OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

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
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("请设置环境变量 OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

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
        system_prompt = load_prompt("rerank_system")
        user_prompt_template = load_prompt("rerank_user")
        user_prompt = user_prompt_template.format(query=question, context=context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
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
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("请设置环境变量 OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

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
        system_prompt = load_prompt("rerank_system")
        user_prompt_template = load_prompt("rerank_user")
        user_prompt = user_prompt_template.format(query=question, context=context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
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
        api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("SILICONFLOW_API_KEY", ""))
        if not api_key:
            raise ValueError("请设置环境变量 OPENAI_API_KEY 或 SILICONFLOW_API_KEY")
        self.client = OpenAI(
            base_url = "https://api.siliconflow.cn/v1",
            api_key = api_key
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

        system_prompt = load_prompt("rerank_system")
        user_prompt_template = load_prompt("rerank_user")
        user_prompt = user_prompt_template.format(query=question, context=context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
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
            api_key = ""
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

        system_prompt = load_prompt("rerank_system")
        user_prompt_template = load_prompt("rerank_user")
        user_prompt = user_prompt_template.format(query=question, context=context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
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
        api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
        if not api_key:
            raise ValueError("请设置环境变量 OPENAI_API_KEY 或 OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url = "https://openrouter.ai/api/v1",
            api_key = api_key
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

        system_prompt = load_prompt("rerank_system")
        user_prompt_template = load_prompt("rerank_user")
        user_prompt = user_prompt_template.format(query=question, context=context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
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
