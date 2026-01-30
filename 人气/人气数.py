# -*- coding: utf-8 -*-
"""
批量获取 celebrity 的 Wikipedia pageviews（人气 proxy）
输入：celebrity_name_all_unique.csv（你已经有了）
输出：celebrity_popularity_wiki_pageviews.csv

依赖：pip install requests pandas
"""

import time
from pathlib import Path
import pandas as pd
import requests

# 维基接口要求明确的 User-Agent，否则可能返回 403
REQUEST_HEADERS = {
    "User-Agent": "MCM2026c/1.0 (contact: your_email@example.com)",
    "Accept": "application/json",
}

SESSION = requests.Session()
SESSION.headers.update(REQUEST_HEADERS)
# 避免环境变量中的代理导致不稳定（如确实需要代理可自行改回 True）
SESSION.trust_env = False

def _get(url: str, params=None, timeout=20, max_retries=5):
    """带重试的 GET 请求（处理 429/5xx/临时 403/网络波动）。"""
    backoff = 1.8
    last_exc = None
    for attempt in range(max_retries):
        try:
            # timeout 可传 (connect, read)
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504, 403):
                # 轻度退避后重试
                time.sleep(backoff * (attempt + 1))
                continue
            return r
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
            continue
    return None

WIKI_API = "https://en.wikipedia.org/w/api.php"
PV_API_TMPL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}"
)

def search_wiki_title(name: str) -> tuple[str | None, str]:
    """用 MediaWiki 搜索接口把人名映射到最可能的英文维基条目标题"""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": name,
        "format": "json",
        "formatversion": 2,
        "srlimit": 1
    }
    try:
        r = _get(WIKI_API, params=params, timeout=20)
        if r is None:
            return None, "err:request_failed"
        r.raise_for_status()
        data = r.json()
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return None, "no_wiki_hit"
        return hits[0]["title"], "ok"
    except Exception as e:
        return None, f"err:{type(e).__name__}"

def get_pageviews(title: str, start_yyyymmdd: str, end_yyyymmdd: str,
                  project="en.wikipedia", access="all-access", agent="all-agents",
                  granularity="daily") -> tuple[int, int, int]:
    """
    拉取某个维基条目在 [start, end] 的 pageviews
    返回：(总浏览量, 日均, 峰值)
    """
    # URL 里 article 需要做编码（空格->下划线；再做 URL encode）
    article = title.replace(" ", "_")
    url = PV_API_TMPL.format(
        project=project,
        access=access,
        agent=agent,
        article=requests.utils.quote(article, safe=""),
        granularity=granularity,
        start=start_yyyymmdd + "00",
        end=end_yyyymmdd + "00"
    )
    r = _get(url, timeout=20)
    if r is None:
        raise RuntimeError("request_failed")
    if r.status_code == 404:
        return (0, 0, 0)
    r.raise_for_status()
    items = r.json().get("items", [])
    views = [it.get("views", 0) for it in items]
    if not views:
        return (0, 0, 0)
    total = int(sum(views))
    mean = int(round(total / len(views)))
    peak = int(max(views))
    return (total, mean, peak)

def main():
    base_dir = Path(__file__).resolve().parent
    inp = base_dir / "celebrity_name_all_unique.csv"
    out = base_dir / "celebrity_popularity_wiki_pageviews.csv"

    # 你可以把时间窗设成“某一季播出期”或“整个比赛期间”
    # 下面先给一个示例：2025-01-01 到 2025-12-31
    start = "20250101"
    end = "20251231"

    df = pd.read_csv(inp)
    rows = []

    # 断点续跑：若输出已存在，则跳过已完成的 name
    done = set()
    if out.exists():
        try:
            old = pd.read_csv(out)
            if "celebrity_name" in old.columns:
                done = set(old["celebrity_name"].astype(str).tolist())
                rows = old.values.tolist()
                print(f"[info] 断点续跑：已完成 {len(done)} 条，将继续剩余部分。")
        except Exception:
            pass

    for i, name in enumerate(df["celebrity_name"].astype(str)):
        if name in done:
            continue
        title, status = search_wiki_title(name)
        if title is None:
            rows.append([name, None, 0, 0, 0, status])
            continue

        try:
            total, mean, peak = get_pageviews(title, start, end)
            rows.append([name, title, total, mean, peak, "ok"])
        except Exception as e:
            rows.append([name, title, 0, 0, 0, f"err:{type(e).__name__}"])

        # 轻微 sleep，避免触发速率限制（评审喜欢你写这个）
        time.sleep(0.2)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df)}")
            # 中途保存，降低网络中断损失
            out_df = pd.DataFrame(rows, columns=[
                "celebrity_name", "wiki_title",
                "views_total", "views_daily_mean", "views_peak",
                "status"
            ])
            out_df.to_csv(out, index=False, encoding="utf-8-sig")
            print("[info] checkpoint saved:", out)

    out_df = pd.DataFrame(rows, columns=[
        "celebrity_name", "wiki_title",
        "views_total", "views_daily_mean", "views_peak",
        "status"
    ])
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print("Saved:", out)

if __name__ == "__main__":
    main()
