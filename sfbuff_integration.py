import asyncio
import html
import json
import os
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

import aiohttp

DEFAULT_SITE_BASE_URL = "https://www.sfbuff.site"
DEFAULT_SITE_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64)"
DEFAULT_TIMEOUT = 10.0


def get_config():
    base_url = os.getenv("SFBUFF_SITE_BASE_URL")
    if not base_url:
        base_url = os.getenv("SFBUFF_API_BASE_URL", DEFAULT_SITE_BASE_URL)
    user_agent = os.getenv("SFBUFF_SITE_USER_AGENT", DEFAULT_SITE_USER_AGENT)
    timeout = float(os.getenv("SFBUFF_API_TIMEOUT", str(DEFAULT_TIMEOUT)))
    return base_url, user_agent, timeout


class SimpleTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.in_caption = False
        self.current_table = None
        self.current_row = None
        self.current_cell = []
        self.current_caption = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.current_table = {"caption": "", "rows": []}
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.current_row = []
        elif self.in_row and tag in ("td", "th"):
            self.in_cell = True
            self.current_cell = []
        elif self.in_table and tag == "caption":
            self.in_caption = True
            self.current_caption = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            cell_text = html.unescape("".join(self.current_cell))
            cell_text = re.sub(r"\s+", " ", cell_text).strip()
            if self.current_row is not None:
                self.current_row.append(cell_text)
        elif tag == "tr" and self.in_row:
            self.in_row = False
            if self.current_table is not None and self.current_row:
                self.current_table["rows"].append(self.current_row)
            self.current_row = None
        elif tag == "caption" and self.in_caption:
            self.in_caption = False
            caption_text = html.unescape("".join(self.current_caption))
            caption_text = re.sub(r"\s+", " ", caption_text).strip()
            if self.current_table is not None:
                self.current_table["caption"] = caption_text
            self.current_caption = []
        elif tag == "table" and self.in_table:
            self.in_table = False
            if self.current_table is not None:
                self.tables.append(self.current_table)
            self.current_table = None

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)
        elif self.in_caption:
            self.current_caption.append(data)


def is_configured():
    base_url, _, _ = get_config()
    return bool(base_url)


def parse_html_tables(html_text):
    parser = SimpleTableParser()
    parser.feed(html_text)
    return parser.tables


def extract_csrf_token(html_text):
    match = re.search(r'name="csrf-token" content="([^"]+)"', html_text)
    return match.group(1) if match else None


def extract_search_uuid(html_text):
    match = re.search(r"fighter_searches/([a-f0-9-]+)", html_text)
    if match:
        return match.group(1)
    match = re.search(r"fighter_search_([a-f0-9-]+)", html_text)
    if match:
        return match.group(1)
    return None


def parse_optional_int(value):
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9-]", "", str(value))
    if cleaned in ("", "-"):
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_ratio(value):
    if value is None:
        return None
    cleaned = str(value).replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_search_results_from_tables(tables):
    for table in tables:
        rows = [row for row in table.get("rows", []) if len(row) >= 6]
        if not rows:
            continue
        data_rows = rows
        if rows and len(rows[0]) >= 2 and not rows[0][1].isdigit():
            data_rows = rows[1:]
        results = []
        for row in data_rows:
            if len(row) < 7:
                continue
            fighter_id = row[0]
            short_id = parse_optional_int(row[1])
            home_country = row[2]
            last_play_at = row[3]
            favorite_character = row[4]
            master_rating = parse_optional_int(row[5])
            league_point = parse_optional_int(row[6])
            results.append({
                "fighter_id": fighter_id,
                "short_id": short_id or row[1],
                "home_country": home_country,
                "last_play_at": last_play_at,
                "favorite_character": favorite_character,
                "master_rating": master_rating,
                "league_point": league_point,
            })
        if results:
            return results
    return []


def parse_rivals_from_tables(tables):
    rivals = {"favorites": [], "victims": [], "tormentors": []}
    for table in tables:
        caption = table.get("caption", "").lower()
        if "favorite" in caption:
            bucket = "favorites"
        elif "victim" in caption:
            bucket = "victims"
        elif "tormentor" in caption:
            bucket = "tormentors"
        else:
            continue
        rows = [row for row in table.get("rows", []) if len(row) >= 8]
        if rows and len(rows[0]) >= 4 and parse_optional_int(rows[0][3]) is None:
            rows = rows[1:]
        for row in rows:
            if len(row) < 8:
                continue
            score = {
                "total": parse_optional_int(row[3]),
                "wins": parse_optional_int(row[4]),
                "losses": parse_optional_int(row[5]),
                "draws": parse_optional_int(row[6]),
                "diff": parse_optional_int(row[7]),
                "ratio": parse_ratio(row[8]) if len(row) > 8 else None,
            }
            rivals[bucket].append({
                "name": row[0],
                "character": row[1],
                "input_type": row[2],
                "score": score,
            })
    return rivals


def parse_matchups_from_tables(tables):
    for table in tables:
        rows = [row for row in table.get("rows", []) if len(row) >= 7]
        if not rows:
            continue
        if rows and len(rows[0]) >= 3 and parse_optional_int(rows[0][2]) is None:
            rows = rows[1:]
        matchups = []
        for row in rows:
            if len(row) < 7:
                continue
            if row[0] in {"Σ", "S", "Sum", "Total"}:
                continue
            score = {
                "total": parse_optional_int(row[2]),
                "wins": parse_optional_int(row[3]),
                "losses": parse_optional_int(row[4]),
                "draws": parse_optional_int(row[5]),
                "diff": parse_optional_int(row[6]),
                "ratio": parse_ratio(row[7]) if len(row) > 7 else None,
            }
            matchups.append({
                "away_character": row[0],
                "away_input_type": row[1],
                "score": score,
            })
        if matchups:
            return matchups
    return []


def parse_matches_from_tables(tables):
    for table in tables:
        rows = [row for row in table.get("rows", []) if len(row) >= 9]
        if not rows:
            continue
        if rows and len(rows[0]) >= 4 and "result" in rows[0][3].lower():
            rows = rows[1:]
        matches = []
        for row in rows:
            if len(row) < 9:
                continue
            matches.append({
                "played_at": row[9] if len(row) > 9 else row[-1],
                "away": {
                    "name": row[4],
                    "character": row[5],
                },
                "result": {"name": row[3]},
            })
        if matches:
            return matches
    return []


def extract_chart_data(html_text):
    match = re.search(r'data-chartjs-data-value="([^"]+)"', html_text)
    if not match:
        return None
    raw = html.unescape(match.group(1))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def parse_ranked_history_from_chart(chart_data):
    if not chart_data:
        return []
    datasets = chart_data.get("data", {}).get("datasets", [])
    mr_dataset = None
    for dataset in datasets:
        if dataset.get("label") == "MR":
            mr_dataset = dataset
            break
    if not mr_dataset:
        return []
    history = []
    for item in mr_dataset.get("data", []):
        label = item.get("label") or ""
        variation = None
        match = re.search(r"\(([-+]?\d+)\)", label)
        if match:
            try:
                variation = int(match.group(1))
            except ValueError:
                variation = None
        history.append({
            "played_at": item.get("x"),
            "mr": item.get("y"),
            "mr_variation": variation,
        })
    return history


def _client_error_payload(exc):
    return {
        "_error": True,
        "status": "client_error",
        "body": str(exc),
    }


async def sfbuff_site_request(session, method, path, params=None, headers=None, data=None, accept=None):
    base_url, user_agent, _ = get_config()
    base_url = base_url.rstrip("/") + "/"
    url = urljoin(base_url, path.lstrip("/"))
    request_headers = {
        "User-Agent": user_agent,
        "Accept-Language": "en",
    }
    if accept:
        request_headers["Accept"] = accept
    if headers:
        request_headers.update(headers)
    async with session.request(method, url, params=params, data=data, headers=request_headers) as response:
        return response.status, await response.text()


async def search(query):
    try:
        _, _, timeout_value = get_config()
        timeout = aiohttp.ClientTimeout(total=timeout_value)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            status, html_text = await sfbuff_site_request(
                session,
                "GET",
                "/fighters/search",
                params={"q": query},
                accept="text/html",
            )
            if status >= 400:
                return {"_error": True, "status": status, "body": html_text}
            csrf_token = extract_csrf_token(html_text)
            if not csrf_token:
                return {"_error": True, "status": "csrf_missing", "body": "csrf token missing"}

            headers = {
                "X-CSRF-Token": csrf_token,
                "X-Requested-With": "XMLHttpRequest",
            }
            status, html_text = await sfbuff_site_request(
                session,
                "POST",
                "/fighter_searches",
                params={"query": query},
                headers=headers,
                accept="text/html",
            )
            if status >= 400:
                return {"_error": True, "status": status, "body": html_text}

            uuid = extract_search_uuid(html_text)
            if not uuid:
                return {"_error": True, "status": "uuid_missing", "body": "uuid missing"}

            for _ in range(6):
                status, html_text = await sfbuff_site_request(
                    session,
                    "GET",
                    f"/fighter_searches/{uuid}",
                    accept="text/vnd.turbo-stream.html",
                )
                if status == 202:
                    await asyncio.sleep(1.5)
                    continue
                if status >= 400:
                    return {"_error": True, "status": status, "body": html_text}
                tables = parse_html_tables(html_text)
                results = parse_search_results_from_tables(tables)
                return {
                    "finished": True,
                    "uuid": uuid,
                    "result": results,
                }

            return {
                "finished": False,
                "uuid": uuid,
                "result": [],
            }
    except aiohttp.ClientError as exc:
        return _client_error_payload(exc)


async def search_status(uuid):
    try:
        _, _, timeout_value = get_config()
        timeout = aiohttp.ClientTimeout(total=timeout_value)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            status, html_text = await sfbuff_site_request(
                session,
                "GET",
                f"/fighter_searches/{uuid}",
                accept="text/vnd.turbo-stream.html",
            )
            if status == 202:
                return {"finished": False, "uuid": uuid, "result": []}
            if status >= 400:
                return {"_error": True, "status": status, "body": html_text}
            tables = parse_html_tables(html_text)
            results = parse_search_results_from_tables(tables)
            return {"finished": True, "uuid": uuid, "result": results}
    except aiohttp.ClientError as exc:
        return _client_error_payload(exc)


async def sync(fighter_id):
    try:
        _, _, timeout_value = get_config()
        timeout = aiohttp.ClientTimeout(total=timeout_value)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            status, html_text = await sfbuff_site_request(
                session,
                "GET",
                f"/fighters/{fighter_id}/matches",
                accept="text/html",
            )
            if status >= 400:
                return {"_error": True, "status": status, "body": html_text}
            csrf_token = extract_csrf_token(html_text)
            if not csrf_token:
                return {"_error": True, "status": "csrf_missing", "body": "csrf token missing"}
            headers = {
                "X-CSRF-Token": csrf_token,
                "X-Requested-With": "XMLHttpRequest",
            }
            status, html_text = await sfbuff_site_request(
                session,
                "POST",
                f"/fighters/{fighter_id}/synchronization",
                headers=headers,
                accept="text/html",
            )
            if status >= 400:
                return {"_error": True, "status": status, "body": html_text}
            return {"started": True}
    except aiohttp.ClientError as exc:
        return _client_error_payload(exc)


async def rivals(fighter_id):
    try:
        _, _, timeout_value = get_config()
        timeout = aiohttp.ClientTimeout(total=timeout_value)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            status, html_text = await sfbuff_site_request(
                session,
                "GET",
                f"/fighters/{fighter_id}/rivals",
                accept="text/html",
            )
            if status >= 400:
                return {"_error": True, "status": status, "body": html_text}
            tables = parse_html_tables(html_text)
            return parse_rivals_from_tables(tables)
    except aiohttp.ClientError as exc:
        return _client_error_payload(exc)


async def matchups(fighter_id):
    try:
        _, _, timeout_value = get_config()
        timeout = aiohttp.ClientTimeout(total=timeout_value)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            status, html_text = await sfbuff_site_request(
                session,
                "GET",
                f"/fighters/{fighter_id}/matchup_chart",
                accept="text/html",
            )
            if status >= 400:
                return {"_error": True, "status": status, "body": html_text}
            tables = parse_html_tables(html_text)
            return parse_matchups_from_tables(tables)
    except aiohttp.ClientError as exc:
        return _client_error_payload(exc)


async def matches(fighter_id):
    try:
        _, _, timeout_value = get_config()
        timeout = aiohttp.ClientTimeout(total=timeout_value)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            status, html_text = await sfbuff_site_request(
                session,
                "GET",
                f"/fighters/{fighter_id}/matches",
                accept="text/html",
            )
            if status >= 400:
                return {"_error": True, "status": status, "body": html_text}
            tables = parse_html_tables(html_text)
            return parse_matches_from_tables(tables)
    except aiohttp.ClientError as exc:
        return _client_error_payload(exc)


async def history(fighter_id):
    try:
        _, _, timeout_value = get_config()
        timeout = aiohttp.ClientTimeout(total=timeout_value)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            status, html_text = await sfbuff_site_request(
                session,
                "GET",
                f"/fighters/{fighter_id}/ranked_history",
                accept="text/html",
            )
            if status >= 400:
                return {"_error": True, "status": status, "body": html_text}
            chart_data = extract_chart_data(html_text)
            return parse_ranked_history_from_chart(chart_data)
    except aiohttp.ClientError as exc:
        return _client_error_payload(exc)
