import json
import random
import math
import os
import time
from openai import OpenAI
from tqdm import tqdm

MAX_DESC_LEN = 3000

ANNOTATION_HF_PROMPT_TEMPLATE = """
You are an annotator tasked with constructing developer intent descriptions for pretrained models.
Each intent should describe what a developer would search for if they wanted to find a model with similar functionality.
You will be given the model’s name and functional description. Based on this information, compose a concise intent description that reflects the developer’s search goal rather than the model’s internal details.

# Instructions
1. Perspective:
* Write from the viewpoint of a developer searching for a model.
* Focus on what the developer wants to achieve or implement, not how the model is built.
* Imagine what kind of query a developer would type into a search engine or repository.

2. Conciseness:
* Avoid redundant or vague terms such as “advanced,” “powerful,” or “state-of-the-art” unless necessary.
* Avoid repeating the model name.
* Keep the intent natural and in plain English.

3. Relevance:
* The intent must closely match the functionality described in the given metadata.
* Do not introduce unrelated capabilities.

# Example
Input:
Model Name: dslim/bert-base-NER
Functional Description: bert-base-NER is a fine-tuned BERT model that is ready to use for Named Entity Recognition and achieves state-of-the-art performance for the NER task. It has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER) and miscellaneous (MISC).

Output:
Intent Description: I want to extract the names of people and places in English text.

# Annotation
Input:
Model Name: {name}
Functional Description: {functional_description}

Output:
Intent Description:
"""

ANNOTATION_JS_PROMPT_TEMPLATE = """
You are an annotator tasked with constructing developer intent descriptions for JavaScript packages.
Each intent should describe what a developer would search for if they wanted to find a package with similar functionality.
You will be given the package’s name and functional description. Based on this information, compose a concise intent description that reflects the developer’s search goal rather than the package’s internal details.

# Instructions
1. Perspective:
* Write from the viewpoint of a developer searching for a package.
* Focus on what the developer wants to achieve or implement, not how the package is built.
* Imagine what kind of query a developer would type into a search engine or repository.

2. Conciseness:
* Avoid redundant or vague terms such as “advanced,” “powerful,” or “state-of-the-art” unless necessary.
* Avoid repeating the package name.
* Keep the intent natural and developer-oriented.

3. Relevance:
* The intent must closely match the functionality described in the given metadata.
* Do not introduce unrelated capabilities.

# Example
Input:
Package Name: React-lazyload
Functional Description: Lazyload your components, images or anything that affects performance.

Output:
Intent Description: I want my website to load faster and only display images when users scroll to them.

# Annotation
Input:
Package Name: {name}
Functional Description: {functional_description}

Output:
Intent Description:
"""

ANNOTATION_LINUX_PROMPT_TEMPLATE = """
You are an annotator tasked with constructing developer intent descriptions for Linux software groups.
Each intent should describe what a developer would search for if they wanted to find a group with similar functionality.
You will be given the group’s name and functional description. Based on this information, compose a concise intent description that reflects the developer’s search goal rather than the group’s internal details.

# Instructions
1. Perspective:
* Write from the viewpoint of a developer searching for a Linux package group.
* Focus on what the developer wants to achieve or enable, not how the group is built.
* Imagine what kind of query a developer would type into a package manager or documentation site.

2. Conciseness:
* Avoid redundant or vague terms such as “advanced,” “powerful,” or “state-of-the-art” unless necessary.
* Avoid repeating the group name.
* Keep the intent natural and focused on usage.

3. Relevance:
* The intent must closely match the functionality described in the given metadata.
* Do not introduce unrelated capabilities.

# Example
Input:
Group Name: Fonts
Functional Description: Fonts for rendering text in a variety of languages and scripts.

Output:
Intent Description: I want my system interface to display text correctly in multiple languages.

# Annotation
Input:
Group Name: {name}
Functional Description: {functional_description}

Output:
Intent Description:
"""

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        all_model = json.load(file)
    return all_model

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def gpt_completion(prompt):
    client = OpenAI(
        api_key="sk-BwTI1iSg83soUQ6u2d1096B8A27848E5B3E4141154Dc592b",  # replace with your key
        base_url="http://66.206.9.230:4000/v1",
    )
    flag = False
    while not flag:
        try:
            response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-4.1-2025-04-14",
                    temperature=0   
            )
            flag=True
        except Exception as e:
            print(e)
            time.sleep(0.5)
    return response.choices[0].message.content


def annotation_hf_model():
    model_list = load_json("IntentRecBench/data/hf/sampled.json")
    model_desc_list = load_json("IntentRecBench/data/hf/desc/all_desc.json")
    annotation_list = []
    for model in tqdm(model_list, desc="Annotation HF Model"):
        model_name = model["name"]
        model_desc = model_desc_list[model_name]
         # ====== 加入防止 prompt 过长的逻辑 ======
        # 限制描述长度在 3000 字符以内（可根据需要调整）
        if len(model_desc) > MAX_DESC_LEN:
            model_desc = model_desc[:MAX_DESC_LEN] + " ... (truncated)"
        annotation = gpt_completion(ANNOTATION_HF_PROMPT_TEMPLATE.format(name=model_name, functional_description=model_desc))
        annotation_list.append({
            "intent": annotation,
            "artifact": model_name
        })
    save_json(annotation_list, "IntentRecBench/data/hf/dataset.json")

def annotation_js_package():
    package_list = load_json("IntentRecBench/data/js/sampled.json")
    package_desc_list = load_json("IntentRecBench/data/js/candidate_artifacts.json")
    annotation_list = []
    for package in tqdm(package_list, desc="Annotation JS Package"):
        package_name = package["name"]
        package_desc = ""
        for item in package_desc_list: 
            if item["name"] == package_name:
                package_desc = item["description"]
                break
        if package_desc == "":
            print(f"Package {package_name} not found in package_desc_list")
            continue
        if len(package_desc) > MAX_DESC_LEN:
            package_desc = package_desc[:MAX_DESC_LEN] + " ... (truncated)"
        annotation = gpt_completion(ANNOTATION_JS_PROMPT_TEMPLATE.format(name=package_name, functional_description=package_desc))
        annotation_list.append({
            "intent": annotation,
            "artifact": package_name,
        })
    save_json(annotation_list, "IntentRecBench/data/js/dataset.json")

def annotation_linux_group():
    group_list = load_json("IntentRecBench/data/linux/candidate_artifacts.json")
    annotation_list = []
    for group in tqdm(group_list, desc="Annotation Linux Group"):
        group_name = group["name"]
        group_desc = group["description"]
        if len(group_desc) > MAX_DESC_LEN:
            group_desc = group_desc[:MAX_DESC_LEN] + " ... (truncated)"
        annotation = gpt_completion(ANNOTATION_LINUX_PROMPT_TEMPLATE.format(name=group_name, functional_description=group_desc))
        annotation_list.append({
            "intent": annotation,
            "artifact": group_name,
        })
    save_json(annotation_list, "IntentRecBench/data/linux/dataset.json")

if __name__ == "__main__":
    # annotation_js_package()
    annotation_linux_group()