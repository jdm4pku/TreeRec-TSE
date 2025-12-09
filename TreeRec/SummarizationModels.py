
import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .utils import load_prompt

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
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("请设置环境变量 OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)

            system_prompt = load_prompt("summarization_system")
            user_prompt_template = load_prompt("summarization_user")
            user_prompt = user_prompt_template.format(context=context)
            
            response = client.chat.completions.create(
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
            api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("SILICONFLOW_API_KEY", ""))
            if not api_key:
                raise ValueError("请设置环境变量 OPENAI_API_KEY 或 SILICONFLOW_API_KEY")
            client = OpenAI(
                base_url = "https://api.siliconflow.cn/v1",
                api_key = api_key
            )

            system_prompt = load_prompt("summarization_system")
            user_prompt_template = load_prompt("summarization_user")
            user_prompt = user_prompt_template.format(context=context)
            
            response = client.chat.completions.create(
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
            api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("SILICONFLOW_API_KEY", ""))
            if not api_key:
                raise ValueError("请设置环境变量 OPENAI_API_KEY 或 SILICONFLOW_API_KEY")
            client = OpenAI(
                base_url = "https://api.siliconflow.cn/v1",
                api_key = api_key
            )

            system_prompt = load_prompt("summarization_system")
            user_prompt_template = load_prompt("summarization_user")
            user_prompt = user_prompt_template.format(context=context)
            
            response = client.chat.completions.create(
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
            api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
            if not api_key:
                raise ValueError("请设置环境变量 OPENAI_API_KEY 或 OPENROUTER_API_KEY")
            client = OpenAI(
                base_url = "https://openrouter.ai/api/v1",
                api_key = api_key
            )

            system_prompt = load_prompt("summarization_system")
            user_prompt_template = load_prompt("summarization_user")
            user_prompt = user_prompt_template.format(context=context)
            
            response = client.chat.completions.create(
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
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e