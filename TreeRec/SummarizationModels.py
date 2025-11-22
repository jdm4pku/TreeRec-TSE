
import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT4oSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-4o-2024-05-13"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in software engineering and requirements modeling. Your task is to analyze multiple software artifacts and identify their shared features."
                            "Your task is to analyze multiple software artifacts and identify their shared features."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The following are names and descriptions of several software artifacts.\n"
                            f"Please read all of them carefully and write a concise, coherent summary "
                            f"that captures their **common characteristics, goals, and functional focus**. "
                            f"Do not list them one by one; instead, generalize their similarities "
                            f"to express the overarching purpose or theme that unites them.\n\n{context}"
                        ),
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


# class GPT3SummarizationModel(BaseSummarizationModel):
#     def __init__(self, model="text-davinci-003"):

#         self.model = model

#     @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
#     def summarize(self, context, max_tokens=500, stop_sequence=None):

#         try:
#             client = OpenAI()

#             response = client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {
#                         "role": "user",
#                         "content": f"Write a summary of the following, including as many key details as possible: {context}:",
#                     },
#                 ],
#                 max_tokens=max_tokens,
#             )

#             return response.choices[0].message.content

#         except Exception as e:
#             print(e)
#             return e

class QwenSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="Qwen/Qwen3-14B"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI(
                base_url = "https://api.siliconflow.cn/v1",
                api_key = "sk-wbnxvocaaofhilzlgkvhiuhoivdawabyvaavkvblnokomdyz"
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in software engineering and requirements modeling. Your task is to analyze multiple software artifacts and identify their shared features."
                            "Your task is to analyze multiple software artifacts and identify their shared features."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The following are names and descriptions of several software artifacts.\n"
                            f"Please read all of them carefully and write a concise, coherent summary "
                            f"that captures their **common characteristics, goals, and functional focus**. "
                            f"Do not list them one by one; instead, generalize their similarities "
                            f"to express the overarching purpose or theme that unites them.\n\n{context}"
                        ),
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e
    
class DeepSeekSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="Pro/deepseek-ai/DeepSeek-R1"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI(
                base_url = "https://api.siliconflow.cn/v1",
                api_key = "sk-wbnxvocaaofhilzlgkvhiuhoivdawabyvaavkvblnokomdyz"
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in software engineering and requirements modeling. Your task is to analyze multiple software artifacts and identify their shared features."
                            "Your task is to analyze multiple software artifacts and identify their shared features."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The following are names and descriptions of several software artifacts.\n"
                            f"Please read all of them carefully and write a concise, coherent summary "
                            f"that captures their **common characteristics, goals, and functional focus**. "
                            f"Do not list them one by one; instead, generalize their similarities "
                            f"to express the overarching purpose or theme that unites them.\n\n{context}"
                        ),
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e

class LlamaSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="meta-llama/llama-3.1-8b-instruct"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI(
                base_url = "https://openrouter.ai/api/v1",
                api_key = "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"
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in software engineering and requirements modeling. Your task is to analyze multiple software artifacts and identify their shared features."
                            "Your task is to analyze multiple software artifacts and identify their shared features."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The following are names and descriptions of several software artifacts.\n"
                            f"Please read all of them carefully and write a concise, coherent summary "
                            f"that captures their **common characteristics, goals, and functional focus**. "
                            f"Do not list them one by one; instead, generalize their similarities "
                            f"to express the overarching purpose or theme that unites them.\n\n{context}"
                        ),
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e