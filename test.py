import requests
import json
from bs4 import BeautifulSoup
import time
import re
import os

BASE = "https://openreview.net"
GROUP_URL = "https://api2.openreview.net/notes"
PROFILE_API = "https://api2.openreview.net/profiles/search"

OUTPUT = "iclr2026_reviewers.json"
SAVE_EVERY = 100   # 每 100 条保存一次

# ============================================================
# 全局请求头（防止被当成机器人封禁）
# ============================================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_INTERVAL = 3     # 每个请求之后 sleep
RETRY_WAIT_429 = 60      # 遇到 429（限流）等待 60 秒
MAX_RETRY = 10           # 最多重试 10 次


# ============================================================
# 安全请求：自动重试、限流处理、避免 JSONDecodeError
# ============================================================
def safe_json_request(method, url, **kwargs):
    kwargs["headers"] = HEADERS

    for attempt in range(MAX_RETRY):
        try:
            r = requests.request(method, url, timeout=20, **kwargs)

            # ---------- 正常返回 ----------
            if r.status_code == 200:
                try:
                    data = r.json()
                    time.sleep(REQUEST_INTERVAL)
                    return data
                except Exception:
                    print("⚠️ JSON parse failed, retrying...")
                    time.sleep(REQUEST_INTERVAL)
                    continue

            # ---------- 限流 ----------
            elif r.status_code == 429:
                print(f"⛔ 429 Rate Limit → waiting {RETRY_WAIT_429}s before retry...")
                time.sleep(RETRY_WAIT_429)
                continue

            # ---------- 其他错误 ----------
            else:
                print(f"⚠️ Bad status={r.status_code}, retrying {attempt+1}/{MAX_RETRY} ...")
                time.sleep(REQUEST_INTERVAL)
                continue

        except Exception as e:
            print(f"⚠️ Request error: {e}, retrying {attempt+1}/{MAX_RETRY} ...")
            time.sleep(REQUEST_INTERVAL)
            continue

    print("❌ Failed too many times → returning empty dict.")
    return {}


# ============================================================
# STEP 1：获取所有 submission（更稳）
# ============================================================
def get_all_submissions():
    submissions = []
    offset = 0
    limit = 1000

    while True:
        payload = {
            "invitation": "ICLR.cc/2026/Conference/-/Submission",
            "limit": limit,
            "offset": offset
        }

        print(f"Fetching submissions offset={offset} ...")
        data = safe_json_request("POST", GROUP_URL, json=payload)

        if not data or "notes" not in data:
            print("⚠️ Invalid response, stopping submission fetch.")
            break

        batch = data["notes"]
        if not batch:
            break

        submissions.extend(batch)
        offset += limit

    print(f"Total submissions fetched: {len(submissions)}")
    return submissions


# ============================================================
# STEP 2：解析论文页面，抽取 Submission Number + reviewer id
# ============================================================
def parse_paper_page(forum_id):
    url = f"{BASE}/forum?id={forum_id}"
    print(f"  -> Fetching paper page: {url}")

    try:
        html = requests.get(url, headers=HEADERS, timeout=20).text
    except Exception:
        print("  !! Failed to load HTML")
        return {"forum_id": forum_id, "submission_number": None, "reviewers": [], "api_results": []}

    soup = BeautifulSoup(html, "html.parser")

    result = {
        "forum_id": forum_id,
        "submission_number": None,
        "reviewers": [],
        "api_results": []
    }

    # 提取 Submission Number
    text = soup.get_text(" ", strip=True)
    m = re.search(r"Submission Number:\s*(\d+)", text)
    if m:
        result["submission_number"] = m.group(1)
    else:
        print("  !! No submission number found")

    # 提取 Reviewer IDs
    reviewer_texts = soup.find_all(text=re.compile(r"Reviewer "))
    for item in reviewer_texts:
        m = re.search(r"Reviewer\s+([A-Za-z0-9]+)", item)
        if m:
            result["reviewers"].append(m.group(1))

    return result


# ============================================================
# STEP 3：请求 reviewer profile API
# ============================================================
def fetch_profile_api(sub_num, reviewer_id):
    group = f"ICLR.cc/2026/Conference/Submission{sub_num}/Reviewer_{reviewer_id}"
    params = {"group": group}
    print(f"       -> Querying reviewer API id={reviewer_id}")
    return safe_json_request("GET", PROFILE_API, params=params)


# ============================================================
# 保存
# ============================================================
def save_progress(data, filename=OUTPUT):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved progress ({len(data)} items).")


# ============================================================
# 主函数流程
# ============================================================
def main():
    print("===== Fetching submission list =====")
    submissions = get_all_submissions()

    # 断点续爬
    if os.path.exists(OUTPUT):
        with open(OUTPUT, "r", encoding="utf-8") as f:
            output = json.load(f)
        done = {item["forum_id"] for item in output}
        print(f"Resuming from previous progress: {len(output)} completed.")
    else:
        output = []
        done = set()

    total = len(submissions)

    for sub in submissions:
        forum_id = sub["id"]

        if forum_id in done:
            continue

        print(f"\n=== [{len(output)+1}/{total}] Processing forum_id={forum_id} ===")

        info = parse_paper_page(forum_id)
        sub_num = info["submission_number"]

        if sub_num:
            for rid in info["reviewers"]:
                api_json = fetch_profile_api(sub_num, rid)
                info["api_results"].append({"reviewer_id": rid, "api_json": api_json})

        output.append(info)

        # 每100篇保存
        if len(output) % SAVE_EVERY == 0:
            save_progress(output)

    # 最终保存
    save_progress(output)
    print("\n🎉 ALL DONE!")


if __name__ == "__main__":
    main()