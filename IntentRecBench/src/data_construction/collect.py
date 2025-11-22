
import os
import time
import requests
from lxml import html
from tqdm import tqdm
import random
import sys
import json
from bs4 import BeautifulSoup

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        all_model = json.load(file)
    if not isinstance(all_model, dict):
        raise ValueError("The input file should be a json file")
    return all_model

def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def count_articles(tree):
    """统计页面中 <article> 数量"""
    articles = tree.xpath("//article")
    return len(articles)


def get_element_value(url, all_model):
    """从单个 Hugging Face 页面提取模型名与类型"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"HTTP Error: {response.status_code}"
    except Exception as e:
        return None, f"Request Error: {e}"

    try:
        tree = html.fromstring(response.content)
        a_counts = count_articles(tree)
    except Exception as e:
        return None, f"HTML Parse Error: {e}"

    if a_counts == 0:
        debug_path = f"IntentRecBench/data/hf/debug/empty_page_{int(time.time())}.html"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        return None, f"No <article> found. Saved to {debug_path}"

    progress_bar = tqdm(total=a_counts, desc=f"Processing {url}", unit="article")
    success_count = 0
    failed_texts = []

    # 遍历所有 article 节点（每页只请求一次）
    articles = tree.xpath("//article")

    for idx, article in enumerate(articles, 1):
        try:
            text_content = article.text_content().strip()
            if not text_content:
                failed_texts.append(f"第{idx}个模型：空文本")
                continue

            first_part = text_content.split("•")[0].strip()
            lines = [line.strip() for line in first_part.split("\n") if line.strip()]

            if len(lines) >= 2:
                model_name = lines[0]
                model_type = lines[1]
            elif len(lines) == 1:
                model_name = lines[0]
                model_type = "unknown"
            else:
                failed_texts.append(f"第{idx}个模型：无法解析 '{text_content[:50]}'")
                continue

            if model_name:
                all_model[model_name] = model_type
                success_count += 1
            else:
                failed_texts.append(f"第{idx}个模型：模型名为空 '{text_content[:50]}'")

        except Exception as e:
            failed_texts.append(f"第{idx}个模型解析错误: {e}")
        finally:
            progress_bar.update(1)

    progress_bar.close()
    print(f"✅ 成功解析 {success_count} 个模型，❌ 失败 {len(failed_texts)} 个")

    # 保存失败信息
    if failed_texts:
        os.makedirs("IntentRecBench/data/hf/debug", exist_ok=True)
        debug_file = f"IntentRecBench/data/hf/debug/failed_parsing_{int(time.time())}.txt"
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n总数: {a_counts}, 成功: {success_count}, 失败: {len(failed_texts)}\n\n")
            f.write("\n".join(failed_texts))
        print(f"⚠️ 解析失败详情已保存到: {debug_file}")

    return success_count, None

def merge_hf_pages():
    """合并所有页面数据并打印重复模型出现在哪几页"""
    import glob
    import re

    all_model = {}
    model_origin = {}  # 记录每个模型第一次出现在哪一页
    duplicates = {}    # 记录重复出现的模型和对应页

    page_dir = "IntentRecBench/data/hf/pages"

    if not os.path.exists(page_dir):
        print("页面目录不存在")
        return

    page_files = glob.glob(os.path.join(page_dir, "page_*.json"))
    # 提取页码并排序
    page_files.sort(key=lambda x: int(re.search(r"page_(\d+)\.json", x).group(1)))

    print(f"找到 {len(page_files)} 个页面文件")

    for page_file in page_files:
        try:
            page_data = load_json(page_file)
            page_name = os.path.basename(page_file)
            before_count = len(all_model)

            for model_name, model_type in page_data.items():
                if model_name in all_model:
                    # 发现重复，记录来源页
                    if model_name not in duplicates:
                        duplicates[model_name] = [model_origin[model_name], page_name]
                    else:
                        if page_name not in duplicates[model_name]:
                            duplicates[model_name].append(page_name)
                else:
                    model_origin[model_name] = page_name
                    all_model[model_name] = model_type

            added_count = len(all_model) - before_count
            print(f"✅ 已合并 {page_name}: 新增 {added_count} 个模型 (当前总数: {len(all_model)})")

        except Exception as e:
            print(f"❌ 合并 {os.path.basename(page_file)} 时出错: {e}")

    # 保存最终结果
    save_json(all_model, "IntentRecBench/data/hf/candidate_name.json")
    print(f"\n🎯 所有页面数据已合并，共 {len(all_model)} 个模型")

    # 打印重复信息
    if duplicates:
        print(f"\n⚠️ 发现 {len(duplicates)} 个重复模型，示例:")
        sample_count = 0
        for model_name, pages in duplicates.items():
            print(f"  - 模型: {model_name}")
            print(f"    出现在页面: {', '.join(pages)}")
            sample_count += 1
            if sample_count >= 10:
                print("    ...（仅展示前10个）")
                break

        # 保存到 debug 文件
        os.makedirs("IntentRecBench/data/hf/debug", exist_ok=True)
        dup_file = "IntentRecBench/data/hf/debug/duplicate_models.txt"
        with open(dup_file, "w", encoding="utf-8") as f:
            f.write(f"共 {len(duplicates)} 个重复模型\n\n")
            for model_name, pages in duplicates.items():
                f.write(f"{model_name}: {', '.join(pages)}\n")

        print(f"🔍 重复模型详情已保存到 {dup_file}")

    else:
        print("✅ 未发现重复模型。")

def delete_code_block(input_str):
    output_str = ""
    label_list = []
    try:
        for i in range(len(input_str)-2):
            if input_str[i: i+3] == '```':
                label_list.append(i)
        for i in range(1, len(label_list), 2):
            label_list[i] += 3
        label_list.insert(0, 0)
        label_list.insert(len(label_list), len(input_str))
        start = [label_list[i] for i in range(0, len(label_list), 2)]
        end = [label_list[i] for i in range(1, len(label_list), 2)]
        if len(start) == len(end):
            for i in range(len(start)):
                output_str += input_str[start[i]: end[i]]
    except:
        output_str = input_str
    return output_str

def get_hf_model_name():
    """主函数：遍历 Hugging Face 模型页"""
    os.makedirs("IntentRecBench/data/hf/pages", exist_ok=True)
    all_model = {}

    # 抓取第 0 页
    url_0 = "https://huggingface.co/models?sort=trending"
    count0, error0 = get_element_value(url_0, all_model)
    if error0:
        print(f"默认页下载失败: {error0}")
        return
    print(f"默认页下载完成，共 {count0} 个模型")
    save_json(all_model, "IntentRecBench/data/hf/pages/page_0.json")

    # 抓取后续页
    for i in range(1, 100):
        page_models = {}
        url = f"https://huggingface.co/models?p={i}&sort=trending"
        print(f"\n📄 开始解析第 {i} 页: {url}")

        max_retries = 5
        for attempt in range(max_retries):
            count_i, error_i = get_element_value(url, page_models)
            if error_i:
                print(f"⚠️ 第 {i} 页失败 ({attempt+1}/{max_retries}) 次: {error_i}")
                time.sleep(5)
                continue
            else:
                print(f"✅ 第 {i} 页下载成功，共 {count_i} 个模型")
                break
        else:
            print(f"❌ 第 {i} 页连续失败 {max_retries} 次，跳过。")
            continue

        # 保存当前页
        page_file = f"IntentRecBench/data/hf/pages/page_{i}.json"
        save_json(page_models, page_file)
        print(f"💾 第 {i} 页已保存到 {page_file}")

        all_model.update(page_models)
        # 防封延时
        time.sleep(random.uniform(1.5, 3.5))

    save_json(all_model, "IntentRecBench/data/hf/candidate_name.json")
    print(f"\n🎯 所有页面数据已合并，共 {len(all_model)} 个模型")

def get_hf_model_desc(model_name):
    model_url = "https://huggingface.co/{}".format(model_name)
    response = requests.get(model_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    target_elements = soup.select("body > div > main > div.container.relative.flex.flex-col.md\\:grid.md\\:space-y-0.w-full.md\\:grid-cols-12.md\\:flex-1.md\\:grid-rows-full.space-y-4.md\\:gap-6 > section.pt-8.border-gray-100.md\\:col-span-7.pb-24.relative.break-words.copiable-code-container")
    model_desc = ""
    for element in target_elements:
        temp = element.text.strip().split("\n")
        temp = list(map(lambda x:x.strip(), temp))
        element_list = [i for i in temp if i != ""]
        model_desc += "\n".join(element_list)
    model_desc = delete_code_block(model_desc)
    return model_desc

def collect_hf_artifacts():
    # 如果文件不存在，下载
    if not os.path.exists("IntentRecBench/data/hf/candidate_desc.json"):
        os.makedirs("IntentRecBench/data/hf/debug", exist_ok=True)
        get_hf_model_name()
        merge_hf_pages()
    # 下载所有的页面描述
    page_dir = "IntentRecBench/data/hf/pages"
    desc_dir = "IntentRecBench/data/hf/desc"
    if not os.path.exists(desc_dir):
        os.makedirs(desc_dir, exist_ok=True)
    for i in range(100):
        page_desc = {}
        page_file = f"{page_dir}/page_{i}.json"
        page_data = load_json(page_file)
        for model_name, model_type in page_data.items():
            try:
                model_desc = get_hf_model_desc(model_name)
            except Exception as e:
                model_desc = ""
                print(f"⚠️ 获取 {model_name} 描述失败: {e}")
            page_desc[model_name] = model_desc
            time.sleep(random.uniform(0.4, 1.0))
        out_file = f"{desc_dir}/page_{i}.json"
        save_json(page_desc, f"{desc_dir}/page_{i}.json")
        print(f"💾 已保存页面描述到 {out_file}（{len(page_desc)} 条）")
        # print(f"💾 已保存页面描述到 {f"{desc_dir}/page_{i}.json"}（{len(page_desc)} 条）")
    
    # 合并所有的页面描述
    all_desc = {}
    for i in range(100):
        page_desc_file = f"{desc_dir}/page_{i}.json"
        page_desc = load_json(page_desc_file)
        all_desc.update(page_desc)
    save_json(all_desc, "IntentRecBench/data/hf/candidate_desc.json")
    print(f"🎯 所有页面数据已合并，共 {len(all_desc)} 个模型")

def format_hf_artifacts():
    all_data = []
    with open("IntentRecBench/data/hf/name/all_name.json", "r", encoding="utf-8") as f:
        all_name = json.load(f)
    with open("IntentRecBench/data/hf/desc/all_desc.json", "r", encoding="utf-8") as f:
        all_desc = json.load(f)
    for key, value in all_desc.items():
        all_data.append({
            "name": key,
            "type": all_name[key],
            "description": value
        })
    save_json(all_data, "IntentRecBench/data/hf/candidate_artifacts.json")

def collect_js_package():
    keywords = ["front-end","cli","css","iot","mobile","robotics","back-end","documentation","testing","coverage","frameworks","math"]
    max_len = 1000
    base_url = "https://registry.npmjs.org/-/v1/search"
    size = 250
    delay = 0.5

    os.makedirs("IntentRecBench/data/js/keywords", exist_ok=True)
    all_packages_total = []

    for keyword in keywords:
        print(f"\n🔍 开始爬取 npm 包（关键词：{keyword}）...")
        all_packages = []
        offset = 0
        max_retries = 5

        while len(all_packages) < max_len:
            params = {
                "text": f"keywords:{keyword}",
                "size": size,
                "from": offset
            }

            try:
                res = requests.get(base_url, params=params, timeout=10)
                res.raise_for_status()
                data = res.json()
                results = data.get("objects", [])
                if not results:  # 没数据了，退出循环
                    print(f"⚠️ 关键词 {keyword} 无更多结果，停止。")
                    break

                for item in results:
                    pkg = item["package"]
                    pkg_name = pkg["name"]
                    pkg_desc = pkg.get("description", "")
                    all_packages.append({
                        "name": pkg_name,
                        "description": pkg_desc
                    })
                    all_packages_total.append({
                        "name": pkg_name,
                        "type": keyword,
                        "description": pkg_desc
                    })

                print(f"✅ 已获取 {len(all_packages)} 个包 (offset={offset})")
                offset += size
                time.sleep(delay)

            except Exception as e:
                print(f"⚠️ 请求失败: {e}")
                for i in range(max_retries):
                    print(f"⏳ 第 {i+1}/{max_retries} 次重试中...")
                    time.sleep(5)
                    try:
                        res = requests.get(base_url, params=params, timeout=10)
                        res.raise_for_status()
                        break
                    except:
                        continue
                else:
                    print(f"❌ 关键词 {keyword} 重试超过上限，跳过。")
                    break

        # 写入当前关键词文件
        output_file = f"IntentRecBench/data/js/keywords/{keyword}.json"
        save_json(all_packages, output_file)
        print(f"📦 已保存 {len(all_packages)} 个包到 {output_file}")

    # 写入总的文件
    total_file = "IntentRecBench/data/js/candidate_artifacts.json"
    save_json(all_packages_total, total_file)
    print(f"\n🎯 所有关键词数据已合并保存到 {total_file}，共 {len(all_packages_total)} 个包。")

if __name__ == "__main__":
    collect_hf_artifacts()
    format_hf_artifacts()
    collect_js_package()