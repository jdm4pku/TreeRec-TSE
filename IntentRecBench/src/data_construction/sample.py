import json
import random
import math
import os

def load_json(file_path):
    with open(file_path,"r") as file:
        all_model = json.load(file)
    return all_model

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def sample_hf_model(radio):
    with open("IntentRecBench/data/hf/name/all_name.json", "r", encoding="utf-8") as f:
        model2type = json.load(f)
    with open("IntentRecBench/data/hf/candidate_artifacts.json", "r", encoding="utf-8") as f:
        model2desc = json.load(f)
    sampled_model2type = {}
    # 按类型分组模型
    new_type_list = ["Text Generation", "Text-to-Image", 
                     "Image-Text-to-Text", "Text Classification", 
                     "Sentence Similarity", "Automatic Speech Recognition", 
                     "Text-to-Speech", "Feature Extraction", "Image-to-Image", 
                     "Image Classification", "Image-to-Text", "Translation", 
                     "Token Classification", "Text-to-Video", 
                     "Zero-Shot Image Classification", "Image Segmentation", 
                     "Object Detection", "Image Feature Extraction", 
                     "Any-to-Any", "Time Series Forecasting", 
                     "Text Ranking", "Image-to-3D", "Audio Classification", 
                     "Depth Estimation", "Keypoint Detection", "Text-to-Audio", 
                     "Zero-shot Classification", "Visual Question Answering", 
                     "Robotics", "Video-Text-to-Text", "Video Classification", 
                     "Uncoditional Image Generation", "Question Answering", 
                     "Zero-shot Object Detection", "Mask Generation", 
                     "Voice Activity Detection", "Visual Document Retrieval", 
                     "Text-to-3D", "Summarization", "Reinforcement Learning",
                     "Document Question Answering", "Tabular Classification", "Tabular Regression", "other"]
    type2models = {}
    for model_name, model_type in model2type.items():
        if model_type not in new_type_list:
            model_type = "other"
        type2models.setdefault(model_type, []).append(model_name)

    # 按每个类型包含的模型数量从大到小排序
    sorted_types = sorted(type2models.items(), key=lambda x: len(x[1]), reverse=True)

    print("📊 每个类型包含的模型数量（从大到小）：")
    for model_type, models in sorted_types:
        print(f"  - {model_type}: {len(models)} 个模型")    
    
    # 对每个type按照比例采样
    remain_type = ["Image-to-3D", "Audio Classification", 
                     "Depth Estimation", "Keypoint Detection", "Text-to-Audio", "Zero-shot Classification", "Visual Question Answering", "Robotics", "Video-Text-to-Text", "Video Classification", 
                     "Uncoditional Image Generation", "Question Answering", 
                     "Zero-shot Object Detection", "Mask Generation", "Voice Activity Detection", "Visual Document Retrieval", "Text-to-3D", "Summarization", "Reinforcement Learning", "Document Question Answering", "Tabular Classification", "Tabular Regression"]
    for model_type, models in sorted_types:
        if model_type == "other":
            continue
        elif model_type not in remain_type:
            n = max(1, math.floor(len(models) * radio))
            sampled = models[:n]
            sampled_model2type[model_type] = sampled
        else:
            sampled_model2type[model_type] = models
    sum = 0
    print("📊 每个类型包含的模型数量（采样后）：")
    for model_type, models in sampled_model2type.items():
        sum += len(models)
        print(f"  - {model_type}: {len(models)} 个模型")
    print(f"📊 采样后总的模型数量：{sum} 个模型")

    sampled_models = []
    for model_type, models in sampled_model2type.items():
        for model in models:
            sampled_models.append(
                {"name": model}
            )
    random.shuffle(sampled_models)
    save_json(sampled_models, "IntentRecBench/data/hf/sampled.json")

def sample_js_package(radio):
    # 打印每个keyword包含的包的数量
    base_dir = "IntentRecBench/data/js/keywords"
    keywords = ["front-end", "cli", "css", "iot", "mobile", "robotics", 
                "back-end", "documentation", "testing", "coverage", 
                "frameworks", "math"]
    keyword2packages = {}
    for kw in keywords:
        file_path = os.path.join(base_dir, f"{kw}.json")
        if not os.path.exists(file_path):
            print(f"⚠️ 未找到文件：{file_path}")
            continue
        data = load_json(file_path)
        keyword2packages[kw] = data
        print(f"✅ {kw}: {len(data)} 个包")
    sum = 0    
    print("\n📊 每个关键词包含的包数量（原始）：")
    for kw, pkgs in keyword2packages.items():
        sum += len(pkgs)
        print(f"  - {kw}: {len(pkgs)}")
    print(f"📊 原始总包数量：{sum}")
    

    # 按照比例对每个keyword的包进行采样
    sampled_keyword2packages = {}
    for kw, pkgs in keyword2packages.items():
        n = max(1, math.floor(len(pkgs) * radio))
        sampled = pkgs[:n]
        sampled_keyword2packages[kw] = sampled

    # 打印每个keyword包含的包的数量（采样后）
    print("\n📊 每个关键词包含的包数量（采样后）：")
    total = 0
    for kw, pkgs in sampled_keyword2packages.items():
        print(f"  - {kw}: {len(pkgs)}")
        total += len(pkgs)
    print(f"📊 采样后总包数量：{total}")

    # 合并并 shuffle
    all_sampled = []
    for kw, pkgs in sampled_keyword2packages.items():
        for pkg in pkgs:
            all_sampled.append({
                "name": pkg["name"],
            })
    random.shuffle(all_sampled)

    # 保存到json文件
    save_json(all_sampled, "IntentRecBench/data/js/sampled.json")
    print("💾 已保存采样结果到 IntentRecBench/data/js/sampled.json")

if __name__ == "__main__":
    # sample_hf_model(0.1)
    sample_js_package(0.1)
    
