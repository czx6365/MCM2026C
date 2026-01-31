# -*- coding: utf-8 -*-
"""
并发版：批量获取 celebrity 的 Wikipedia pageviews（人气 proxy）
输入：celebrity_name_all_unique.csv
输出：celebrity_popularity_wiki_pageviews.csv
依赖：pip install requests pandas
"""

import csv
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

# ====== 基本配置 ======
REQUEST_HEADERS = {
    "User-Agent": "MCM2026c/1.0 (contact: your_email@example.com)",
    "Accept": "application/json",
}

WIKI_API = "https://en.wikipedia.org/w/api.php"
PV_API_TMPL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}"
)

# ====== 线程本地 Session（每个线程一个，避免共享 Session 的潜在问题） ======
_thread_local = threading.local()

def get_session() -> requests.Session:
    """为每个线程创建独立 Session，并挂载连接池，提高 keep-alive 利用率。"""
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(REQUEST_HEADERS)
        sess.trust_env = False  # 避免环境代理干扰

        # 连接池：并发时显著减少握手/连接开销
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=0)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)

        _thread_local.session = sess
    return sess

def _get(url: str, params=None, timeout=(6, 20), max_retries=5):
    """带退避重试的 GET（处理 429/5xx/临时 403/网络波动）。"""
    backoff = 1.8
    sess = get_session()
    last_exc = None

    for attempt in range(max_retries):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504, 403):
                time.sleep(backoff * (attempt + 1))
                continue
            return r
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
            continue
    return None

def search_wiki_title(name: str):
    """用 MediaWiki search 接口将人名映射到最可能的英文维基标题。"""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": name,
        "format": "json",
        "formatversion": 2,
        "srlimit": 1
    }
    try:
        r = _get(WIKI_API, params=params)
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
                  granularity="monthly"):
    """
    拉取某个维基条目在 [start, end] 的 pageviews
    返回：(总浏览量, 均值, 峰值)
    注：granularity 建议 monthly（更快、更稳），除非你必须要日级波动。
    """
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

    r = _get(url)
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

def worker_one(name: str, start: str, end: str, granularity: str):
    """单条任务：search title -> fetch pageviews"""
    title, status = search_wiki_title(name)
    if title is None:
        return [name, None, 0, 0, 0, status]

    try:
        total, mean, peak = get_pageviews(title, start, end, granularity=granularity)
        return [name, title, total, mean, peak, "ok"]
    except Exception as e:
        return [name, title, 0, 0, 0, f"err:{type(e).__name__}"]

def main():
    base_dir = Path(__file__).resolve().parent
    inp = base_dir / "celebrity_name_all_unique.csv"
    out = base_dir / "celebrity_popularity_wiki_pageviews.csv"

    # 时间窗：示例 2025-01-01 到 2025-12-31
    start = "20250101"
    end = "20251231"

    # granularity 建议 monthly（快很多）
    granularity = "monthly"  # 如必须日级：改成 "daily"

    df = pd.read_csv(inp)
    names = df["celebrity_name"].astype(str).tolist()

    # 断点续跑：读取已完成项
    done = set()
    if out.exists():
        try:
            old = pd.read_csv(out)
            if "celebrity_name" in old.columns:
                done = set(old["celebrity_name"].astype(str).tolist())
                print(f"[info] 断点续跑：已完成 {len(done)} 条，将继续剩余部分。")
        except Exception:
            pass

    todo = [n for n in names if n not in done]
    if not todo:
        print("[info] 全部已完成，无需运行。")
        return

    # 如果文件不存在，先写表头；存在则追加写
    need_header = not out.exists()
    f = open(out, "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(f)
    if need_header:
        writer.writerow([
            "celebrity_name", "wiki_title",
            "views_total", "views_mean", "views_peak",
            "status"
        ])

    # 并发数：建议 8~16，根据网络/稳定性调整
    max_workers = 12

    completed = 0
    t0 = time.time()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(worker_one, name, start, end, granularity): name
                for name in todo
            }

            for fut in as_completed(futures):
                row = fut.result()
                writer.writerow(row)  # 追加写入，避免频繁重写整文件
                completed += 1

                # 每隔一段强制 flush，降低中断损失
                if completed % 50 == 0:
                    f.flush()
                    elapsed = time.time() - t0
                    print(f"[info] done {completed}/{len(todo)} | elapsed {elapsed:.1f}s")

    finally:
        f.flush()
        f.close()

    print("[info] Saved:", out)

if __name__ == "__main__":
    main()
