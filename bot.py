import discord
import os
import asyncio
import base64
import random
import datetime
import re
import mimetypes
import aiohttp
import csv
import pandas as pd
from dotenv import load_dotenv
from aiohttp import web
import sfbuff_integration

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
try:
    CHANNEL_ID = int(os.getenv('CHANNEL_ID'))
except (TypeError, ValueError):
    print("Error: CHANNEL_ID not found or invalid in .env")
    CHANNEL_ID = None

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'xiaomi/mimo-v2-flash:free')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
OPENROUTER_ENABLED = bool(OPENROUTER_API_KEY)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview')
GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'
GEMINI_ENABLED = bool(GEMINI_API_KEY)
GEMINI_INLINE_MAX_BYTES = int(os.getenv('GEMINI_INLINE_MAX_BYTES', '10485760'))
GEMINI_THINKING_LEVEL = os.getenv('GEMINI_THINKING_LEVEL', 'high')
GEMINI_IMAGE_RESOLUTION = os.getenv('GEMINI_IMAGE_RESOLUTION', 'media_resolution_high')
GEMINI_VIDEO_RESOLUTION = os.getenv('GEMINI_VIDEO_RESOLUTION', 'media_resolution_low')
MEDIA_HISTORY_LIMIT = int(os.getenv('MEDIA_HISTORY_LIMIT', '6'))
GEMINI_MEDIA_RESOLUTION_ENABLED = "v1alpha" in GEMINI_BASE_URL.lower()

LLM_ENABLED = GEMINI_ENABLED or OPENROUTER_ENABLED

if GEMINI_ENABLED:
    print(f"Gemini enabled with model: {GEMINI_MODEL}")
elif OPENROUTER_ENABLED:
    print(f"OpenRouter enabled with model: {OPENROUTER_MODEL}")
else:
    print("No LLM API key found. Chat feature disabled.")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)

NEXT_RUN_TIME = None

SYSTEM_PROMPT = (
    "You are Chinese Bub, a wise and eccentric sensei with an undying love for Chairman Mao, kpop, fighting games (especially Street Fighter), and Chinese history. "
    "You deeply care about your friends and want them to IMPROVE in everything they do - especially fighting games! "
    "You MUST respond by FABRICATING creative connections between Mao/China and the topic. MAKE UP fake history/facts. "
    "Example: If discussing Street Fighter, claim Mao invented the Hadouken. "
    "OCCASIONALLY mention some of these figures: {selected_figures_str}. "
    "Make up outlandish fake achievements for them. "
    "Be confident, passionate, and completely make things up! "
    "8pm pst is a legendary time where mystical events happen. Occasionally mention it. "
    "Never break character. Stay in persona always. "
    "Focus on ONE single topic or story per response. Do not ramble or stray off topic. "
    "Do not end responses with a question unless necessary. Keep it casual and natural. "
    "Always speak the same language as the prompt. you are an english speaker by default unless prompted otherwise."
    "5 sentence limit. Keep it punchy. "
    "Answer the user's question DIRECTLY first. "
    "No Tangents. Stay on topic. Keep responses concise and relevant."
    "NEVER output your internal thought process. Do not use parentheses for meta-commentary."
    "If MEDIA_CONTEXT is present and viewable=true, explicitly acknowledge the media and mention one concrete visual detail in your first sentence. "
    "If MEDIA_CONTEXT is present and viewable=false, state you cannot view the media and ask for a brief description. "
    "Never claim to see media unless viewable=true. "
    "When discussing Street Fighter 6 frame data, ONLY use the data provided in 'AVAILABLE DATA' sections. Do not invent or guess frame data values."
    "When discussing broader street fighter topics, use the data provided in 'AVAILABLE DATA' sections. Do not invent or guess frame data values."
)

MOVE_DEFINITIONS = (
    "Glossary:\n"
    "- LK, LP, L in move names = Light moves (Fast but weak)\n"
    "- MK, MP, M in move names = Medium moves (Balanced)\n"
    "- HK, HP, H in move names = Heavy moves (Slow but strong)\n"
    "- OD in move names = Overdrive/EX moves (Enhanced versions, cost meter)\n"
    "- Drive/Super Data: DDoH (Drive Dmg Hit), DDoB (Drive Dmg Block), SelfSoH (Super Gain Hit), SelfSoB (Super Gain Block)\n"
)

IMPROVEMENT_PROMPT = (
    "You are Chinese Bub, a wise sensei who is OBSESSED with improvement and self-betterment. "
    "Your friends are warriors, and you believe they MUST grind, train, and level up - especially in fighting games like Street Fighter 6! "
    "You are extremely passionate and encouraging. "
    "Someone just replied to your question about improvement. Respond about their improvement journey. "
    "Give them motivation! Hype them up! Reference training, frame data, combos, ranked matches, or life skills. "
    "You can tie improvement to Chairman Mao's revolutionary spirit or Chinese resilience. "
    "OCCASIONALLY mention some of these figures: {selected_figures_str}. Make up outlandish fake achievements for them. "
    "Be supportive but also playfully demand more from them. "
    "Never break character. "
    "Focus on ONE single topic or story per response. Do not ramble or stray off topic. "
    "Do not end responses with a question unless necessary. Keep it casual and natural. "
    "Always speak the same language as the prompt. you are an english speaker by default unless prompted otherwise. "
    "5 sentence limit. Keep it punchy. "
    "Answer the user's message DIRECTLY first. "
    "No Tangents. Stay on topic. Keep responses concise and relevant. "
    "NEVER output your internal thought process. Do not use parentheses for meta-commentary. "
    "If MEDIA_CONTEXT is present and viewable=true, explicitly acknowledge the media and mention one concrete visual detail in your first sentence. "
    "If MEDIA_CONTEXT is present and viewable=false, state you cannot view the media and ask for a brief description. "
    "Never claim to see media unless viewable=true. "
    "When discussing Street Fighter 6 frame data, ONLY use the data provided in 'AVAILABLE DATA' sections. Do not invent or guess frame data values. "
    "When discussing broader street fighter topics, use the data provided in 'AVAILABLE DATA' sections. Do not invent or guess frame data values."
)


async def get_openrouter_response(messages):
    """Call OpenRouter API and return the response text."""
    if not OPENROUTER_ENABLED:
        raise RuntimeError("OpenRouter not configured")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://discord.com",
        "X-Title": "Chinese Bub Bot"
    }
    
    clean_messages = [
        {"role": msg.get("role", ""), "content": msg.get("content", "")}
        for msg in messages
    ]

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": clean_messages,
        "max_tokens": 4096,
        "temperature": 1.0,
        "top_p": 1.0,
        "reasoning": {
            "effort": "medium"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"OpenRouter API error {response.status}: {error_text}")
            
            data = await response.json()
            content = data['choices'][0]['message']['content']
            
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = re.sub(r'^\s*\(.*?\)\s*', '', content, flags=re.DOTALL)
            
            return content.strip()


def has_llm_content(messages):
    for msg in messages:
        if msg.get("role") == "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            return True
        parts = msg.get("parts")
        if isinstance(parts, list):
            for part in parts:
                text = part.get("text", "") if isinstance(part, dict) else ""
                if isinstance(text, str) and text.strip():
                    return True
                if isinstance(part, dict) and (
                    part.get("inlineData") or part.get("inline_data")
                ):
                    return True
    return False


def collect_embed_urls(msg):
    urls = []
    for embed in msg.embeds:
        if embed.url:
            urls.append(embed.url)
        if embed.thumbnail and embed.thumbnail.url:
            urls.append(embed.thumbnail.url)
        if embed.image and embed.image.url:
            urls.append(embed.image.url)
        if embed.video and embed.video.url:
            urls.append(embed.video.url)
    return urls


async def get_message_media_items(message):
    items = []

    def add_attachment(att):
        items.append({
            "url": att.url,
            "filename": att.filename,
            "content_type": att.content_type,
            "size": att.size,
        })

    for att in message.attachments:
        add_attachment(att)

    for url in collect_embed_urls(message):
        items.append({
            "url": url,
            "filename": os.path.basename(url.split("?")[0]) or "embed",
            "content_type": mimetypes.guess_type(url)[0],
            "size": None,
        })

    if message.reference and message.reference.message_id:
        try:
            if message.reference.cached_message:
                replied_msg = message.reference.cached_message
            else:
                replied_msg = await message.channel.fetch_message(message.reference.message_id)
            if replied_msg:
                for att in replied_msg.attachments:
                    add_attachment(att)
                for url in collect_embed_urls(replied_msg):
                    items.append({
                        "url": url,
                        "filename": os.path.basename(url.split("?")[0]) or "embed",
                        "content_type": mimetypes.guess_type(url)[0],
                        "size": None,
                    })
        except Exception:
            pass

    if not items:
        media_keywords = ["image", "photo", "pic", "picture", "gif", "video", "screenshot"]
        if any(kw in message.content.lower() for kw in media_keywords):
            try:
                async for prev_msg in message.channel.history(
                    limit=MEDIA_HISTORY_LIMIT,
                    before=message,
                ):
                    if prev_msg.author == client.user:
                        continue
                    prev_items = []
                    for att in prev_msg.attachments:
                        prev_items.append({
                            "url": att.url,
                            "filename": att.filename,
                            "content_type": att.content_type,
                            "size": att.size,
                        })
                    for url in collect_embed_urls(prev_msg):
                        prev_items.append({
                            "url": url,
                            "filename": os.path.basename(url.split("?")[0]) or "embed",
                            "content_type": mimetypes.guess_type(url)[0],
                            "size": None,
                        })
                    if prev_items:
                        items.extend(prev_items)
                        break
            except Exception:
                pass

    deduped = {}
    for item in items:
        url = item.get("url")
        if not url:
            continue
        if url not in deduped:
            deduped[url] = item
    return list(deduped.values())


async def build_message_media_parts(items):
    parts = []
    notes = []
    if not items:
        return parts, notes

    async with aiohttp.ClientSession() as session:
        for item in items:
            url = item.get("url")
            if not url:
                continue
            content_type = item.get("content_type") or ""
            filename = item.get("filename") or "media"
            size = item.get("size")
            if not content_type:
                guessed_type = mimetypes.guess_type(filename)[0]
                if not guessed_type:
                    guessed_type = mimetypes.guess_type(url.split("?")[0])[0]
                content_type = guessed_type or ""
            data = None
            final_type = ""
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        notes.append(f"{filename} fetch failed ({response.status})")
                        continue
                    data = await response.read()
                    header_type = response.headers.get("Content-Type", "")
                    final_type = header_type.split(";")[0].strip().lower()
            except Exception as e:
                notes.append(f"{filename} fetch failed ({e})")
                continue

            if not final_type:
                final_type = content_type
            if not final_type.startswith(("image/", "video/")):
                notes.append(f"{filename} unsupported type {final_type or 'unknown'}")
                continue
            if len(data) > GEMINI_INLINE_MAX_BYTES:
                notes.append(f"{filename} too large for inline media")
                continue
            media_resolution = (
                GEMINI_IMAGE_RESOLUTION
                if final_type.startswith("image/")
                else GEMINI_VIDEO_RESOLUTION
            )
            part = {
                "inlineData": {
                    "mimeType": final_type,
                    "data": base64.b64encode(data).decode("ascii"),
                }
            }
            if GEMINI_MEDIA_RESOLUTION_ENABLED:
                part["mediaResolution"] = {"level": media_resolution}
            parts.append(part)
    return parts, notes


def get_media_context(items, media_parts, media_notes):
    if not items:
        return ""
    image_count = 0
    video_count = 0
    other_count = 0
    for item in items:
        content_type = item.get("content_type") or ""
        filename = item.get("filename") or ""
        url = item.get("url") or ""
        if not content_type:
            guessed_type = mimetypes.guess_type(filename)[0]
            if not guessed_type:
                guessed_type = mimetypes.guess_type(url.split("?")[0])[0]
            content_type = guessed_type or ""
        if content_type.startswith("image/"):
            image_count += 1
        elif content_type.startswith("video/"):
            video_count += 1
        else:
            other_count += 1
    viewable = bool(media_parts)
    if not viewable and media_notes:
        viewable = False
    return (
        "MEDIA_CONTEXT: "
        f"images={image_count}, videos={video_count}, other={other_count}, "
        f"viewable={str(viewable).lower()}"
    )


def build_gemini_payload(messages):
    system_parts = []
    contents = []
    def normalize_parts(parts):
        normalized = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text", "")
            if isinstance(text, str) and text.strip():
                normalized.append({"text": text})
                continue
            inline_data = part.get("inlineData") or part.get("inline_data")
            if inline_data:
                if "mimeType" not in inline_data and "mime_type" in inline_data:
                    inline_data = {
                        "mimeType": inline_data.get("mime_type"),
                        "data": inline_data.get("data"),
                    }
                normalized_part = {"inlineData": inline_data}
                media_resolution = (
                    part.get("mediaResolution")
                    or part.get("media_resolution")
                )
                if media_resolution and GEMINI_MEDIA_RESOLUTION_ENABLED:
                    normalized_part["mediaResolution"] = media_resolution
                normalized.append(normalized_part)
                continue
        return normalized
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        parts = msg.get("parts")
        if not content:
            if not parts:
                continue
        if role == "system":
            system_parts.append(content)
            continue
        if role == "assistant":
            gemini_role = "model"
        else:
            gemini_role = "user"
        if parts:
            parts = normalize_parts(parts)
            if parts:
                contents.append({"role": gemini_role, "parts": parts})
        else:
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 4096,
            "temperature": 1.0,
            "topP": 1.0,
        },
    }
    thinking_level = (GEMINI_THINKING_LEVEL or "").strip().lower()
    if thinking_level:
        payload["generationConfig"]["thinkingConfig"] = {
            "thinkingLevel": thinking_level,
        }
    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}]
        }
    if not contents:
        raise RuntimeError("Gemini payload empty: no user/model content")
    return payload


async def get_gemini_response(messages):
    """Call Gemini API and return the response text."""
    if not GEMINI_ENABLED:
        raise RuntimeError("Gemini not configured")
    payload = build_gemini_payload(messages)
    url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Gemini API error {response.status}: {error_text}")
            data = await response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini API error: empty candidates")
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(part.get("text", "") for part in parts)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = re.sub(r'^\s*\(.*?\)\s*', '', content, flags=re.DOTALL)
            return content.strip()


async def get_llm_response(messages):
    """Call the configured LLM provider and return the response text."""
    if not has_llm_content(messages):
        raise RuntimeError("LLM payload empty: no user content")
    if GEMINI_ENABLED:
        return await get_gemini_response(messages)
    if OPENROUTER_ENABLED:
        return await get_openrouter_response(messages)
    raise RuntimeError("No LLM provider configured")


def truncate_message(text, limit=1800):
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def format_score(score):
    if not score:
        return "-"
    wins = score.get("wins") or 0
    losses = score.get("losses") or 0
    draws = score.get("draws") or 0
    diff = score.get("diff")
    ratio = score.get("ratio")
    details = []
    if diff is not None:
        details.append(f"diff {diff}")
    if ratio is not None:
        try:
            ratio_value = float(ratio)
            details.append(f"{ratio_value:.1f}%")
        except (TypeError, ValueError):
            details.append(f"{ratio}%")
    if details:
        return f"{wins}-{losses}-{draws} ({', '.join(details)})"
    return f"{wins}-{losses}-{draws}"


def format_date(date_text):
    if not date_text:
        return "-"
    return str(date_text).split("T")[0]


def format_search_results(results, limit=5):
    if not results:
        return "No fighters found."
    lines = []
    for entry in results[:limit]:
        fighter_id = entry.get("fighter_id", "?")
        short_id = entry.get("short_id", "?")
        character = entry.get("favorite_character", "?")
        mr = entry.get("master_rating")
        lp = entry.get("league_point")
        home = entry.get("home_country", "?")
        mr_text = "-" if mr in (None, 0) else str(mr)
        lp_text = "-" if lp in (None, 0) else str(lp)
        lines.append(f"{fighter_id} | ID {short_id} | {character} | MR {mr_text} LP {lp_text} | {home}")
    return "\n".join(lines)


def format_rivals_section(title, rivals):
    lines = [f"{title}:"]
    if not rivals:
        lines.append("none")
        return "\n".join(lines)
    for idx, rival in enumerate(rivals, start=1):
        name = rival.get("name", "?")
        character = rival.get("character", "?")
        input_type = rival.get("input_type", "?")
        score = format_score(rival.get("score"))
        lines.append(f"{idx}. {name} ({character}, {input_type}) {score}")
    return "\n".join(lines)


def format_matchups(matchups, limit=10):
    if not matchups:
        return "No matchups found."
    def matchup_key(entry):
        score = entry.get("score") or {}
        ratio = score.get("ratio") or 0
        total = score.get("total") or 0
        return (ratio, total)

    sorted_matchups = sorted(matchups, key=matchup_key, reverse=True)
    lines = []
    for entry in sorted_matchups[:limit]:
        character = entry.get("away_character", "?")
        input_type = entry.get("away_input_type", "?")
        score = format_score(entry.get("score"))
        lines.append(f"{character} ({input_type}) {score}")
    return "\n".join(lines)


def format_ranked_history(history, limit=10):
    if not history:
        return "No ranked history found."
    lines = []
    for item in history[-limit:]:
        played_at = format_date(item.get("played_at"))
        mr = item.get("mr")
        variation = item.get("mr_variation")
        if variation is None:
            variation_text = ""
        else:
            variation_text = f" ({variation:+})"
        lines.append(f"{played_at}: MR {mr}{variation_text}")
    return "\n".join(lines)


def format_matches(matches, limit=10):
    if not matches:
        return "No matches found."
    lines = []
    for match in matches[:limit]:
        played_at = format_date(match.get("played_at"))
        away = match.get("away", {})
        away_name = away.get("name", "?")
        away_character = away.get("character", "?")
        result = match.get("result", {}).get("name", "?")
        lines.append(f"{played_at}: vs {away_name} ({away_character}) result {result}")
    return "\n".join(lines)


def cfn_help_text():
    return (
        "CFN commands:\n"
        "cfn search <query>\n"
        "cfn status <uuid>\n"
        "cfn sync <short_id>\n"
        "cfn rivals <short_id>\n"
        "cfn matchup <short_id>\n"
        "cfn history <short_id>\n"
        "cfn matches <short_id>"
    )


def extract_cfn_command(message):
    content = message.content.strip()
    if not content:
        return None
    content_lower = content.lower()
    if client.user and client.user.mentioned_in(message):
        content_no_mentions = re.sub(r"<@!?[0-9]+>", "", content).strip()
        content_no_mentions_lower = content_no_mentions.lower()
        if content_no_mentions_lower.startswith("cfn"):
            return content_no_mentions[3:].strip()
    return None


def is_valid_short_id(value):
    return bool(re.fullmatch(r"\d{9,}", value or ""))


async def handle_cfn_command(message):
    command_text = extract_cfn_command(message)
    if command_text is None:
        return False

    if not sfbuff_integration.is_configured():
        await message.reply("CFN service is not configured.")
        return True

    return await handle_cfn_site_command(message, command_text)


async def handle_cfn_site_command(message, command_text):
    tokens = command_text.split()
    if not tokens:
        await message.reply(cfn_help_text())
        return True

    action = tokens[0].lower()
    if action in {"help", "?"}:
        await message.reply(cfn_help_text())
        return True

    async with message.channel.typing():
        if action == "search":
            if len(tokens) < 2:
                await message.reply("Usage: cfn search <query>")
                return True
            query = " ".join(tokens[1:]).strip()
            payload = await sfbuff_integration.search(query)
            if payload.get("_error"):
                await message.reply(truncate_message(f"CFN search error: {payload.get('body')}"))
                return True
            if payload.get("finished"):
                response = format_search_results(payload.get("result"))
            else:
                response = (
                    f"Search queued (uuid {payload.get('uuid')}). "
                    "Try again in a few seconds with `@chinese bub cfn status <uuid>`."
                )
            await message.reply(truncate_message(response))
            return True

        if action == "status":
            if len(tokens) < 2:
                await message.reply("Usage: cfn status <uuid>")
                return True
            uuid = tokens[1]
            payload = await sfbuff_integration.search_status(uuid)
            if payload.get("_error"):
                await message.reply(truncate_message(f"CFN status error: {payload.get('body')}"))
                return True
            if payload.get("finished"):
                response = format_search_results(payload.get("result"))
            else:
                response = (
                    f"Search still running (uuid {payload.get('uuid')}). "
                    "Try again in a few seconds."
                )
            await message.reply(truncate_message(response))
            return True

        if action in {"sync", "rivals", "matchup", "history", "matches"}:
            if len(tokens) < 2:
                await message.reply(f"Usage: cfn {action} <short_id>")
                return True
            fighter_id = tokens[1]
            if not is_valid_short_id(fighter_id):
                await message.reply("CFN short_id must be at least 9 digits.")
                return True

            if action == "sync":
                payload = await sfbuff_integration.sync(fighter_id)
                if payload.get("_error"):
                    await message.reply(truncate_message(f"CFN sync error: {payload.get('body')}"))
                    return True
                await message.reply(f"Sync started for {fighter_id} on sfbuff.site.")
                return True

            if action == "rivals":
                payload = await sfbuff_integration.rivals(fighter_id)
                if payload.get("_error"):
                    await message.reply(truncate_message(f"CFN rivals error: {payload.get('body')}"))
                    return True
                sections = [
                    format_rivals_section("Favorites", payload.get("favorites")),
                    format_rivals_section("Victims", payload.get("victims")),
                    format_rivals_section("Tormentors", payload.get("tormentors")),
                ]
                await message.reply(truncate_message("\n\n".join(sections)))
                return True

            if action == "matchup":
                payload = await sfbuff_integration.matchups(fighter_id)
                if isinstance(payload, dict) and payload.get("_error"):
                    await message.reply(truncate_message(f"CFN matchup error: {payload.get('body')}"))
                    return True
                response = format_matchups(payload)
                await message.reply(truncate_message(response))
                return True

            if action == "history":
                payload = await sfbuff_integration.history(fighter_id)
                if isinstance(payload, dict) and payload.get("_error"):
                    await message.reply(truncate_message(f"CFN history error: {payload.get('body')}"))
                    return True
                response = format_ranked_history(payload)
                await message.reply(truncate_message(response))
                return True

            if action == "matches":
                payload = await sfbuff_integration.matches(fighter_id)
                if isinstance(payload, dict) and payload.get("_error"):
                    await message.reply(truncate_message(f"CFN matches error: {payload.get('body')}"))
                    return True
                response = format_matches(payload)
                await message.reply(truncate_message(response))
                return True

    await message.reply(cfn_help_text())
    return True


FRAME_DATA = {}
FRAME_STATS = {}
BNB_DATA = {}
OKI_DATA = {}
CHARACTER_INFO = {}

CHARACTER_ALIASES = {
    "kim": "kimberly",
    "gief": "zangief",
    "sim": "dhalsim",
    "chun": "chun-li",
    "dj": "dee jay",
    "deejay": "dee jay",
    "honda": "e.honda",
    "bison": "m.bison",
    "m bison": "m.bison",
    "dictator": "m.bison",
    "viper": "c.viper",
    "c viper": "c.viper",
    "aki": "a.k.i",
    "a.k.i": "a.k.i",
    "a.k.i.": "a.k.i",
}


def normalize_char_name(name: str) -> str:
    """Normalize character name to lowercase alphanumeric."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def format_sheet_text(df: pd.DataFrame) -> str:
    """Convert DataFrame rows to pipe-separated text lines."""
    df = df.fillna("")
    lines = []
    for _, row in df.iterrows():
        values = []
        for val in row.tolist():
            text = str(val).strip()
            if text.lower() == "nan":
                text = ""
            values.append(text)
        while values and values[0] == "":
            values.pop(0)
        while values and values[-1] == "":
            values.pop()
        if not values:
            continue
        lines.append(" | ".join(values))
    return "\n".join(lines)

def load_frame_data():
    """Load frame data, stats, combos, oki, and character info from ODS."""
    global FRAME_DATA, FRAME_STATS, BNB_DATA, OKI_DATA, CHARACTER_INFO
    filename = "FAT - SF6 Frame Data.ods"
    
    if os.path.exists(filename):
        try:
            print(f"Loading ODS file: {filename} (This may take a moment)...")
            # Load the entire workbook
            xls = pd.ExcelFile(filename, engine='odf')
            
            # Iterate through all sheet names
            for sheet_name in xls.sheet_names:
                # Look for sheets ending in "Normal" (e.g. "ManonNormal", "RyuNormal")
                if sheet_name.endswith("Normal"):
                    # Extract character name (ManonNormal -> manon)
                    char_name = sheet_name.replace("Normal", "").rstrip(".").lower()
                    
                    # Parse the sheet
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    
                    # Convert to list of dicts (replace NaN with empty string)
                    records = df.fillna("").to_dict('records')
                    
                    # Inject character name into each record for reverse lookup context
                    for r in records: r['char_name'] = char_name.capitalize()
                    
                    FRAME_DATA[char_name] = records
                    # print(f"Loaded {len(records)} moves for {char_name}")
                
                # Look for sheets ending in "Stats" (e.g. "ManonStats", "RyuStats")
                elif sheet_name.endswith("Stats") and not sheet_name.startswith("_OLD"):
                    char_name = sheet_name.replace("Stats", "").rstrip(".").lower()
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                
                    stats_dict = dict(zip(df['name'], df['stat']))
                    FRAME_STATS[char_name] = stats_dict
                    # print(f"Loaded stats for {char_name}")
                    
            print(f"Total characters loaded: {len(FRAME_DATA)}")
            print(f"Total stats loaded: {len(FRAME_STATS)}")
            
            BNB_DATA = {}
            OKI_DATA = {}
            CHARACTER_INFO = {}
            normalized_chars = {
                normalize_char_name(name): name
                for name in FRAME_DATA.keys()
            }

            combo_sheets = [
                name for name in xls.sheet_names
                if name.lower().endswith(" combos")
            ]
            oki_sheets = [
                name for name in xls.sheet_names
                if name.lower().endswith(" okisetups")
                or name.lower().endswith(" setupsoki")
            ]

            for sheet_name in combo_sheets:
                char_label = sheet_name[:-len(" combos")]
                char_key = normalized_chars.get(normalize_char_name(char_label))
                if not char_key:
                    continue
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=str)
                combo_text = format_sheet_text(df)
                if combo_text:
                    BNB_DATA[char_key] = combo_text

            for sheet_name in oki_sheets:
                suffix = " okisetups" if sheet_name.lower().endswith(" okisetups") else " setupsoki"
                char_label = sheet_name[:-len(suffix)]
                char_key = normalized_chars.get(normalize_char_name(char_label))
                if not char_key:
                    continue
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=str)
                oki_text = format_sheet_text(df)
                if oki_text:
                    OKI_DATA[char_key] = oki_text

            for sheet_name in xls.sheet_names:
                lower_name = sheet_name.lower()
                if lower_name.endswith(" combos"):
                    continue
                if lower_name.endswith(" okisetups") or lower_name.endswith(" setupsoki"):
                    continue
                if lower_name.endswith(" frame data"):
                    continue
                normalized_name = normalize_char_name(sheet_name)
                char_key = normalized_chars.get(normalized_name)
                if not char_key:
                    continue
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=str)
                info_text = format_sheet_text(df)
                if info_text:
                    CHARACTER_INFO[char_key] = info_text

            print(f"Total combo sheets loaded: {len(BNB_DATA)} characters")
            print(f"Total oki sheets loaded: {len(OKI_DATA)} characters")
            print(f"Total character info sheets loaded: {len(CHARACTER_INFO)} characters")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    else:
        print(f"File not found: {filename}")

def find_moves_in_text(text):
    """Extract character/move mentions and return context payload with mode."""
    found_data = []
    text_lower = text.lower()
    text_lower = re.sub(r"<@!?\d+>", " ", text_lower)
    text_lower = re.sub(r"<@&\d+>", " ", text_lower)
    text_lower = re.sub(r"<#\d+>", " ", text_lower)
    text_lower = re.sub(r"\s+", " ", text_lower).strip()
    tc_prompt_blocks = []
    tc_ambiguous_inputs = set()
    text_tokens = re.findall(r"[a-z0-9]+", text_lower)

    def tokens_in_text(needle_tokens):
        if not needle_tokens:
            return False
        if len(needle_tokens) == 1:
            return needle_tokens[0] in text_tokens
        for i in range(len(text_tokens) - len(needle_tokens) + 1):
            if text_tokens[i : i + len(needle_tokens)] == needle_tokens:
                return True
        return False
    
    # 1. Identify which characters are mentioned
    mentioned_chars = []
    
    # First check for character aliases and normalize them
    for alias, canonical in CHARACTER_ALIASES.items():
        alias_tokens = re.findall(r"[a-z0-9]+", alias.lower())
        if tokens_in_text(alias_tokens):
            if canonical in FRAME_DATA and canonical not in mentioned_chars:
                mentioned_chars.append(canonical)
    
    # Then check for direct character name matches
    for char in FRAME_DATA.keys():
        char_tokens = re.findall(r"[a-z0-9]+", char)
        if tokens_in_text(char_tokens) and char not in mentioned_chars:
            mentioned_chars.append(char)
    
    # Check for BNB/Combo requests
    bnb_keywords = ["combo", "combos", "bnb", "bnbs", "bread and butter", "route", "routes"]
    oki_keywords = ["oki", "okizeme", "setup", "setups", "meaty", "meaties"]
    info_keywords = [
        "playstyle",
        "gameplan",
        "archetype",
        "overview",
        "tell me about",
        "who is",
        "strengths",
        "weaknesses",
        "moveset",
        "toolkit",
        "role",
        "how to play",
        "character synopsis",
        "summary",
        "anti air",
        "anti-air",
        "neutral",
        "win condition",
    ]
    frame_keywords = [
        "frame data",
        "framedata",
        "startup",
        "recovery",
        "active",
        "on block",
        "on hit",
        "hitstun",
        "blockstun",
        "frames",
    ]
    comparison_keywords = [
        "which is better",
        "which is faster",
        "compare",
        "comparison",
        "versus",
    ]
    punish_keywords = ["punish", "punishable", "can i punish", "is it punishable"]
    wants_bnb = any(kw in text_lower for kw in bnb_keywords)
    wants_oki = any(kw in text_lower for kw in oki_keywords)
    wants_info = any(kw in text_lower for kw in info_keywords)
    wants_comparison = (
        any(kw in text_lower for kw in comparison_keywords)
        or re.search(r"\bvs\b", text_lower)
    )
    wants_frame_data = any(kw in text_lower for kw in frame_keywords) or any(
        kw in text_lower for kw in punish_keywords
    ) or wants_comparison
    bnb_context = ""
    info_blocks = []
    if wants_bnb or wants_oki:
        for char in mentioned_chars:
            if wants_bnb and char in BNB_DATA:
                bnb_context += f"\n\n**{char.capitalize()} Combos:**\n{BNB_DATA[char]}"
            if (wants_bnb or wants_oki) and char in OKI_DATA:
                bnb_context += f"\n\n**{char.capitalize()} Oki/Setups:**\n{OKI_DATA[char]}"
    results = []
    if wants_frame_data:
        # 2. Heuristic: For each mentioned character, search for moves mentioned nearby?
        # Simpler approach: Check if any move inputs are present in the text
        # that map to these characters.

    
        move_regex = r"\b([1-9][0-9]*[a-zA-Z]+|stand\s+[a-zA-Z]+|crouch\s+[a-zA-Z]+|jump\s+[a-zA-Z]+|[a-zA-Z]+\s+kick|[a-zA-Z]+\s+punch)\b"
        potential_inputs = re.findall(move_regex, text_lower)
        extra_inputs = []
        for token in ["dp", "srk", "shoryu", "shoryuken", "623"]:
            if re.search(rf"\b{re.escape(token)}\b", text_lower):
                extra_inputs.append(token)
        if "sway" in text_lower:
            extra_inputs.append("sway")
        if "jus cool" in text_lower or "juscool" in text_lower:
            extra_inputs.append("jus cool")
        # 46P charge patterns (back-forward+punch)
        charge_patterns = [
            (r"\b46p\b", "46p"),
            (r"\b46lp\b", "46lp"),
            (r"\b46mp\b", "46mp"),
            (r"\b46hp\b", "46hp"),
            (r"\b46pp\b", "46pp"),
            (r"\bb,\s*f\+?p\b", "46p"),
            (r"\bb,\s*f\+?lp\b", "46lp"),
            (r"\bb,\s*f\+?mp\b", "46mp"),
            (r"\bb,\s*f\+?hp\b", "46hp"),
            (r"\bb,\s*f\+?pp\b", "46pp"),
            (r"\bbf\+?p\b", "46p"),
            (r"\bbf\+?lp\b", "46lp"),
            (r"\bbf\+?mp\b", "46mp"),
            (r"\bbf\+?hp\b", "46hp"),
            (r"\bbf\+?pp\b", "46pp"),
            (r"\bback\s*forward\+?p\b", "46p"),
            (r"\bback\s*forward\+?lp\b", "46lp"),
            (r"\bback\s*forward\+?mp\b", "46mp"),
            (r"\bback\s*forward\+?hp\b", "46hp"),
            (r"\bback\s*forward\+?pp\b", "46pp"),
            # 28K charge patterns (down-up+kick)
            (r"\b28k\b", "28k"),
            (r"\b28lk\b", "28lk"),
            (r"\b28mk\b", "28mk"),
            (r"\b28hk\b", "28hk"),
            (r"\b28kk\b", "28kk"),
            (r"\bd,\s*u\+?k\b", "28k"),
            (r"\bd,\s*u\+?lk\b", "28lk"),
            (r"\bd,\s*u\+?mk\b", "28mk"),
            (r"\bd,\s*u\+?hk\b", "28hk"),
            (r"\bd,\s*u\+?kk\b", "28kk"),
            (r"\bdu\+?k\b", "28k"),
            (r"\bdu\+?lk\b", "28lk"),
            (r"\bdu\+?mk\b", "28mk"),
            (r"\bdu\+?hk\b", "28hk"),
            (r"\bdu\+?kk\b", "28kk"),
            (r"\bdown\s*up\+?k\b", "28k"),
            (r"\bdown\s*up\+?lk\b", "28lk"),
            (r"\bdown\s*up\+?mk\b", "28mk"),
            (r"\bdown\s*up\+?hk\b", "28hk"),
            (r"\bdown\s*up\+?kk\b", "28kk"),
        ]
        for pattern, token in charge_patterns:
            if re.search(pattern, text_lower) and token not in extra_inputs:
                extra_inputs.append(token)
        combo_text = text_lower.replace("->", ">")
        tc_present = bool(
            re.search(r"\b(tc|target combo|targetcombo)\b", text_lower)
        )
        combo_matches = re.findall(
            r"\b[0-9a-zA-Z+]+(?:\s*>\s*[0-9a-zA-Z+]+)+\b",
            combo_text,
        )
        for combo in combo_matches:
            combo_token = re.sub(r"\s+", "", combo)
            if combo_token not in extra_inputs:
                extra_inputs.append(combo_token)
        if tc_present and not combo_matches and mentioned_chars:
            compact_text = re.sub(r"\s+", "", combo_text)
            for char in mentioned_chars:
                normalized_char = normalize_char_name(char)
                compact_text_for_char = re.sub(
                    rf"\b{re.escape(normalized_char)}\b", "", compact_text
                )
                if compact_text_for_char == compact_text:
                    compact_text_for_char = compact_text
                tc_map = {}
                for row in FRAME_DATA.get(char, []):
                    num_cmd_raw = str(row.get("numCmd", ""))
                    num_cmd = num_cmd_raw.lower()
                    if ">" not in num_cmd:
                        continue
                    base_cmd = num_cmd.split(">", 1)[0].strip()
                    base_key = re.sub(r"\s+", "", base_cmd)
                    if not base_key:
                        continue
                    tc_map.setdefault(base_key, []).append(num_cmd_raw)
                for base_key, combos in tc_map.items():
                    if base_key not in compact_text_for_char:
                        continue
                    if len(combos) == 1:
                        combo_token = re.sub(r"\s+", "", combos[0].lower())
                        if combo_token not in extra_inputs:
                            extra_inputs.append(combo_token)
                        continue
                    tc_ambiguous_inputs.add(base_key)
                    combo_list = "\n".join(f"- {combo}" for combo in combos)
                    tc_prompt_blocks.append(
                        f"**Target Combo Options ({char.capitalize()})**\n"
                        f"{base_key.upper()} follow-ups:\n{combo_list}\n"
                        "Reply with the exact follow-up numCmd."
                    )
        if tc_present:
            combo_token_pattern = (
                r"(?:[1-9][0-9a-zA-Z+]*|lp|mp|hp|lk|mk|hk|pp|kk|p|k)"
            )
            combo_sequences = re.findall(
                rf"\b{combo_token_pattern}(?:\s+{combo_token_pattern})+\b",
                text_lower,
            )
            for combo in combo_sequences:
                if not re.search(r"\d", combo):
                    continue
                combo_token = re.sub(r"\s+", "", combo)
                if combo_token not in extra_inputs:
                    extra_inputs.append(combo_token)
        keyword_inputs = [
            # 46P moves
            "air slasher",
            "sonic boom",
            "sumo headbutt",
            "psycho crusher",
            "rolling attack",
            "blanka ball",
            "bison crusher",
            "crusher",
            "fireball",
            "boom",
            "headbutt",
            "ball",
            # 28K moves
            "vertical rolling attack",
            "upball",
            "up ball",
            "somersault kick",
            "flash kick",
            "flashkick",
            "shadow rise",
            "command jump",
            "fly",
            "jackknife maximum",
            "upkicks",
            "upkick",
            "up kicks",
            "sumo smash",
            "ass slam",
            "butt slam",
            "spinning bird kick",
            "sbk",
        ]
        # Strength prefixes for charge moves
        strength_prefixes = ["lp", "mp", "hp", "od", "ex", "light", "medium", "heavy", "l", "m", "h"]
        for token in keyword_inputs:
            # Check for strength+keyword combos (e.g., "heavy fireball", "hp boom")
            for prefix in strength_prefixes:
                combo = f"{prefix} {token}"
                if combo in text_lower and combo not in extra_inputs:
                    if token == "ball" and "blanka" not in text_lower:
                        continue
                    extra_inputs.append(combo)
            # Check for bare keyword
            if re.search(rf"\b{re.escape(token)}\b", text_lower) and token not in extra_inputs:
                if token == "ball" and "blanka" not in text_lower:
                    continue
                extra_inputs.append(token)
        potential_inputs.extend(extra_inputs)

        # also valid simple inputs: "mp", "hk" if preceded by char?

        for char in mentioned_chars:
            char_data = FRAME_DATA[char]

            # Check against potential inputs found via regex
            for inp in potential_inputs:
                row = lookup_frame_data(char, inp)
                if row and row not in results:
                    results.append(row)

            # Also check strict "frame data [char] [move]" remainder if exists
            # (This handles the specific verified cases)

            # "brute force" check for short inputs if the regex missed them (like "mp")
            # only if the string looks like "ryu mp"
            if f"{char} mp" in text_lower:
                row = lookup_frame_data(char, "mp")
                if row and row not in results:
                    results.append(row)
            if f"{char} mk" in text_lower:
                row = lookup_frame_data(char, "mk")
                if row and row not in results:
                    results.append(row)
            if f"{char} hp" in text_lower:
                row = lookup_frame_data(char, "hp")
                if row and row not in results:
                    results.append(row)
            if f"{char} hk" in text_lower:
                row = lookup_frame_data(char, "hk")
                if row and row not in results:
                    results.append(row)
            if f"{char} lp" in text_lower:
                row = lookup_frame_data(char, "lp")
                if row and row not in results:
                    results.append(row)
            if f"{char} lk" in text_lower:
                row = lookup_frame_data(char, "lk")
                if row and row not in results:
                    results.append(row)

        # SPD/360 variations - for Zangief (Screw Piledriver) and Lily (Mexican Typhoon)
            spd_patterns = [
                ("l spd", "lp"), ("m spd", "mp"), ("h spd", "hp"),
                ("light spd", "lp"), ("medium spd", "mp"), ("heavy spd", "hp"),
                ("lspd", "lp"), ("mspd", "mp"), ("hspd", "hp"),
                ("od spd", "od"), ("ex spd", "od"),
                ("360+lp", "lp"), ("360+mp", "mp"), ("360+hp", "hp"), ("360+pp", "od"),
                ("360lp", "lp"), ("360mp", "mp"), ("360hp", "hp"), ("360pp", "od"),
                ("spd", ""), ("360", ""),
            ]
            for pattern, strength in spd_patterns:
                if pattern in text_lower:
                    # Try both Screw Piledriver (Gief) and Mexican Typhoon (Lily)
                    if strength:
                        move_names = [f"{strength} screw piledriver", f"{strength} mexican typhoon"]
                    else:
                        move_names = ["screw piledriver", "mexican typhoon"]
                    for move_name in move_names:
                        row = lookup_frame_data(char, move_name)
                        if row and row not in results:
                            results.append(row)
                            break
                    break  # Only match one SPD variant

            # Chun-Li stance/serenity stream and followups
            stance_patterns = [
                ("stance lp", "stance lp"), ("stance mp", "stance mp"), ("stance hp", "stance hp"),
                ("stance lk", "stance lk"), ("stance mk", "stance mk"), ("stance hk", "stance hk"),
                ("ss lp", "ss lp"), ("ss mp", "ss mp"), ("ss hp", "ss hp"),
                ("ss lk", "ss lk"), ("ss mk", "ss mk"), ("ss hk", "ss hk"),
                ("serenity stream", "stance"), ("stance", "stance"), ("ss", "ss"),
            ]
            for pattern, alias_key in stance_patterns:
                if pattern in text_lower:
                    row = lookup_frame_data(char, alias_key)
                    if row and row not in results:
                        results.append(row)
                    break  # Only match one stance variant

            # Lily Mexican Typhoon variations
            typhoon_patterns = [
                ("l typhoon", "l typhoon"), ("m typhoon", "m typhoon"), ("h typhoon", "h typhoon"),
                ("light typhoon", "light typhoon"), ("medium typhoon", "medium typhoon"), ("heavy typhoon", "heavy typhoon"),
                ("od typhoon", "od typhoon"), ("ex typhoon", "ex typhoon"),
                ("mexican typhoon", "mexican typhoon"), ("typhoon", "typhoon"),
            ]
            for pattern, alias_key in typhoon_patterns:
                if pattern in text_lower:
                    row = lookup_frame_data(char, alias_key)
                    if row and row not in results:
                        results.append(row)
                    break  # Only match one typhoon variant

    # Format the results
    formatted_blocks = []
    
    # 3. Add Character Stats if relevant keywords found
    stats_keywords = ["stats", "health", "health", "drive", "reversal", "jump", "dash", "speed", "throw"]
    wants_stats = any(k in text_lower for k in stats_keywords)
    
    if wants_stats:
        for char in mentioned_chars:
            if char in FRAME_STATS:
                s = FRAME_STATS[char]
                # Format specific stats or all of them? 
                # Let's provide the key ones: Health, Best Reversal, Dashes, Jumps
                # The user asked for "best reversal" specifically.
                reversal_name = s.get('bestReversal', '?')
                
                stats_block = (
                    f"**{char.capitalize()} Stats**\n"
                    f"Health: {s.get('health', '?')}\n"
                    f"Best Reversal: {reversal_name}\n"
                    f"Forward Dash: {s.get('fDash', '?')}f // Back Dash: {s.get('bDash', '?')}f\n"
                    f"Jump: {s.get('nJump', '?')}f\n"
                )
                formatted_blocks.append(stats_block)
                
                # RECURSIVE LOOKUP: If we have a best reversal name, fetch its REAL frame data
                # so the LLM doesn't hallucinate it.
                if reversal_name and reversal_name != '?':
                     # Try to find this move in the moves list
                     rev_row = lookup_frame_data(char, str(reversal_name))
                     if rev_row and rev_row not in results:
                         results.append(rev_row)

    # 4. AUTO-INJECT KEY MOVES (Context Injection)
    # If we have a character but NO specific moves found (e.g. "Help me with Ryu"),
    # the LLM will try to give advice about buttons. We MUST provide the data for those likely buttons
    # to prevent hallucinations (like saying 5MK is special cancellable when it isn't).
    if wants_frame_data and mentioned_chars and not results:
        key_moves = ["5MP", "5MK", "2MK", "5HP", "2HP", "5HK", "2HK"]
        for char in mentioned_chars:
            for km in key_moves:
                k_row = lookup_frame_data(char, km)
                if k_row and k_row not in results:
                    results.append(k_row)

    has_results = bool(results)

    if not wants_frame_data and (
        wants_info or (mentioned_chars and not has_results and not wants_bnb and not wants_stats)
    ):
        for char in mentioned_chars:
            if char in CHARACTER_INFO:
                info_blocks.append(
                    f"**{char.capitalize()} Overview:**\n{CHARACTER_INFO[char]}"
                )

    for move_data in results:
        def clean(val):
            return str(val).replace('*', ',')

        startup = clean(move_data.get('startup', '-'))
        active = clean(move_data.get('active', '-'))
        recovery = clean(move_data.get('recovery', '-')).replace('(', ' (Whiff: ')
        cancel = clean(move_data.get('xx', '-'))
        damage = clean(move_data.get('dmg', '-'))
        guard = clean(move_data.get('atkLvl', '-'))
        on_hit = clean(move_data.get('onHit', '-'))
        on_block = clean(move_data.get('onBlock', '-'))
        extra_info = clean(move_data.get('extraInfo', '-')).replace('[', '').replace(']', '').replace('"', '')
        
        # New Stats (Drive/Super)
        ddoh = clean(move_data.get('DDoH', '-'))
        ddob = clean(move_data.get('DDoB', '-'))
        dgain = clean(move_data.get('DGain', '-'))
        ssoh = clean(move_data.get('SelfSoH', '-'))
        ssob = clean(move_data.get('SelfSoB', '-'))
        
        gauge_info = (
             f"Drive Dmg: Hit {ddoh} / Block {ddob} // Drive Gain: {dgain}\n"
             f"Super Gain: Hit {ssoh} / Block {ssob}\n"
        )
        
        # Hit Confirm Data (Conditional)
        wants_hc = any(k in text_lower for k in ['confirm', 'hc', 'window'])
        hc_info = ""
        if wants_hc:
            hc_sp = clean(move_data.get('hcWinSpCa', '-'))
            hc_tc = clean(move_data.get('hcWinTc', '-'))
            hc_notes = clean(move_data.get('hcWinNotes', '')).replace('[', '').replace(']', '').replace('"', '')
            hc_info = (
                f"Hit Confirm (Sp/Su): {hc_sp} // Hit Confirm (TC): {hc_tc}\n"
                f"Hit Confirm Notes: {hc_notes}\n"
            )

        # Stun Data (Always Included)
        hstun = clean(move_data.get('hitstun', '-'))
        bstun = clean(move_data.get('blockstun', '-'))
        stun_info = f"Stun Frames: Hit {hstun} // Block {bstun}\n"

        block = (
            f"**{move_data['moveName']} ({move_data['numCmd']})**\n"
            f"Character: {move_data.get('char_name', 'Unknown')}\n"
            f"Startup: {startup} // Active: {active} // Recovery: {recovery}\n"
            f"Cancel: {cancel}\n"
            f"Damage: {damage}\n"
            f"Guard: {guard}\n"
            f"On Hit: {on_hit} // On Block: {on_block}\n"
            f"{gauge_info}"
            f"{stun_info}"
            f"{hc_info}"
            f"Notes: {extra_info}"
        )
        formatted_blocks.append(block)

    if tc_prompt_blocks:
        formatted_blocks.extend(tc_prompt_blocks)
    
    sections = []
    if formatted_blocks:
        sections.append("\n\n".join(formatted_blocks))
    if info_blocks:
        sections.append("\n\n".join(info_blocks))
    if bnb_context:
        sections.append(bnb_context.strip())

    output = "\n\n---\n".join(sections)

    # Check for punish calculation
    punish_verdict = check_punish(text_lower, results)
    if punish_verdict:
        if output:
            output = punish_verdict + "\n\n---\n\n" + output
        else:
            output = punish_verdict

    has_frame_blocks = bool(formatted_blocks)
    has_combo_blocks = bool(bnb_context)
    has_overview_blocks = bool(info_blocks)
    if has_frame_blocks:
        mode = "frame"
    elif has_combo_blocks:
        mode = "combo"
    elif has_overview_blocks:
        mode = "overview"
    else:
        mode = "none"

    return {
        "data": output,
        "mode": mode,
    }


def lookup_frame_data(character, move_input):
    """Search for a move in character's frame data by numCmd, plnCmd, or moveName."""
    move_input = str(move_input)
    char_key = character.lower()
    if char_key not in FRAME_DATA:
        return None
    
    data = FRAME_DATA[char_key]
    move_input = move_input.lower().strip()
    combo_input = None
    if ">" in move_input or "->" in move_input:
        combo_input = re.sub(r"\s+", "", move_input.replace("->", ">"))
    
    INPUT_ALIASES = {
        # Chun-Li
        "4mp": "4 or 6mp",
        "6mp": "4 or 6mp",
        "f+mp": "4 or 6mp",
        "b+mp": "4 or 6mp",
        "fmp": "4 or 6mp",
        "bmp": "4 or 6mp",
        # DP/SRK
        "dp": "623",
        "srk": "623",
        "shoryu": "623",
        "shoryuken": "623",
        "sway": "juggling sway",
        "juggling sway": "juggling sway",
        "jus cool": "jus cool",
        "juscool": "jus cool",
        # Zangief SPD
        "360": "screw piledriver",
        "spd": "screw piledriver",
        "360p": "screw piledriver",
        "360+lp": "lp screw piledriver",
        "360+mp": "mp screw piledriver",
        "360+hp": "hp screw piledriver",
        "360+pp": "od screw piledriver",
        "360lp": "lp screw piledriver",
        "360mp": "mp screw piledriver",
        "360hp": "hp screw piledriver",
        "360pp": "od screw piledriver",
        "l spd": "lp screw piledriver",
        "m spd": "mp screw piledriver",
        "h spd": "hp screw piledriver",
        "od spd": "od screw piledriver",
        "ex spd": "od screw piledriver",
        "lspd": "lp screw piledriver",
        "mspd": "mp screw piledriver",
        "hspd": "hp screw piledriver",
        "light spd": "lp screw piledriver",
        "medium spd": "mp screw piledriver",
        "heavy spd": "hp screw piledriver",
        # Chun-Li Serenity Stream
        "stance": "serenity stream",
        "ss": "serenity stream",
        "stance lp": "orchid palm",
        "stance mp": "snake strike",
        "stance hp": "lotus fist",
        "stance lk": "forward strike",
        "stance mk": "senpu kick",
        "stance hk": "tenku kick",
        "ss lp": "orchid palm",
        "ss mp": "snake strike",
        "ss hp": "lotus fist",
        "ss lk": "forward strike",
        "ss mk": "senpu kick",
        "ss hk": "tenku kick",
        # Lily Mexican Typhoon
        "typhoon": "mexican typhoon",
        "mexican typhoon": "mexican typhoon",
        "l typhoon": "lp mexican typhoon",
        "m typhoon": "mp mexican typhoon",
        "h typhoon": "hp mexican typhoon",
        "od typhoon": "od mexican typhoon",
        "ex typhoon": "od mexican typhoon",
        "light typhoon": "lp mexican typhoon",
        "medium typhoon": "mp mexican typhoon",
        "heavy typhoon": "hp mexican typhoon",
    }

    CHARACTER_INPUT_ALIASES = {
        "dee jay": {
            "46p": "air slasher",
            "46lp": "lp air slasher",
            "46mp": "mp air slasher",
            "46hp": "hp air slasher",
            "46pp": "od air slasher",
            "air slasher": "air slasher",
            "slasher": "air slasher",
            "fireball": "air slasher",
            "lp fireball": "lp air slasher",
            "mp fireball": "mp air slasher",
            "hp fireball": "hp air slasher",
            "od fireball": "od air slasher",
            "ex fireball": "od air slasher",
            "light fireball": "lp air slasher",
            "medium fireball": "mp air slasher",
            "heavy fireball": "hp air slasher",
            "l fireball": "lp air slasher",
            "m fireball": "mp air slasher",
            "h fireball": "hp air slasher",
            # 28K Jackknife Maximum
            "28k": "jackknife maximum",
            "28lk": "lk jackknife maximum",
            "28mk": "mk jackknife maximum",
            "28hk": "hk jackknife maximum",
            "28kk": "od jackknife maximum",
            "jackknife maximum": "jackknife maximum",
            "jackknife": "jackknife maximum",
            "upkicks": "jackknife maximum",
            "up kicks": "jackknife maximum",
            "flash kick": "jackknife maximum",
            "lk upkicks": "lk jackknife maximum",
            "mk upkicks": "mk jackknife maximum",
            "hk upkicks": "hk jackknife maximum",
            "od upkicks": "od jackknife maximum",
            "light upkicks": "lk jackknife maximum",
            "medium upkicks": "mk jackknife maximum",
            "heavy upkicks": "hk jackknife maximum",
        },
        "guile": {
            "46p": "sonic boom",
            "46lp": "lp sonic boom",
            "46mp": "mp sonic boom",
            "46hp": "hp sonic boom",
            "46pp": "od sonic boom",
            "sonic boom": "sonic boom",
            "boom": "sonic boom",
            "fireball": "sonic boom",
            "lp boom": "lp sonic boom",
            "mp boom": "mp sonic boom",
            "hp boom": "hp sonic boom",
            "od boom": "od sonic boom",
            "ex boom": "od sonic boom",
            "light boom": "lp sonic boom",
            "medium boom": "mp sonic boom",
            "heavy boom": "hp sonic boom",
            "l boom": "lp sonic boom",
            "m boom": "mp sonic boom",
            "h boom": "hp sonic boom",
        },
        "e.honda": {
            "46p": "sumo headbutt",
            "46lp": "lp sumo headbutt",
            "46mp": "mp sumo headbutt",
            "46hp": "hp sumo headbutt",
            "46pp": "od sumo headbutt",
            "sumo headbutt": "sumo headbutt",
            "headbutt": "sumo headbutt",
            "lp headbutt": "lp sumo headbutt",
            "mp headbutt": "mp sumo headbutt",
            "hp headbutt": "hp sumo headbutt",
            "od headbutt": "od sumo headbutt",
            "ex headbutt": "od sumo headbutt",
            "light headbutt": "lp sumo headbutt",
            "medium headbutt": "mp sumo headbutt",
            "heavy headbutt": "hp sumo headbutt",
            "l headbutt": "lp sumo headbutt",
            "m headbutt": "mp sumo headbutt",
            "h headbutt": "hp sumo headbutt",
            # 28K Sumo Smash
            "28k": "sumo smash",
            "28lk": "lk sumo smash",
            "28mk": "mk sumo smash",
            "28hk": "hk sumo smash",
            "28kk": "od sumo smash",
            "sumo smash": "sumo smash",
            "ass slam": "sumo smash",
            "butt slam": "sumo smash",
            "lk sumo smash": "lk sumo smash",
            "mk sumo smash": "mk sumo smash",
            "hk sumo smash": "hk sumo smash",
            "od sumo smash": "od sumo smash",
            "light ass slam": "lk sumo smash",
            "medium ass slam": "mk sumo smash",
            "heavy ass slam": "hk sumo smash",
        },
        "m.bison": {
            "46p": "psycho crusher",
            "46lp": "lp psycho crusher",
            "46mp": "mp psycho crusher",
            "46hp": "hp psycho crusher",
            "46pp": "od psycho crusher",
            "psycho crusher": "psycho crusher",
            "bison crusher": "psycho crusher",
            "crusher": "psycho crusher",
            "lp crusher": "lp psycho crusher",
            "mp crusher": "mp psycho crusher",
            "hp crusher": "hp psycho crusher",
            "od crusher": "od psycho crusher",
            "ex crusher": "od psycho crusher",
            "light crusher": "lp psycho crusher",
            "medium crusher": "mp psycho crusher",
            "heavy crusher": "hp psycho crusher",
            "l crusher": "lp psycho crusher",
            "m crusher": "mp psycho crusher",
            "h crusher": "hp psycho crusher",
            # 28K Shadow Rise
            "28k": "shadow rise",
            "28kk": "od shadow rise",
            "shadow rise": "shadow rise",
            "command jump": "shadow rise",
            "fly": "shadow rise",
            "od shadow rise": "od shadow rise",
        },
        "blanka": {
            # 46P Rolling Attack
            "46p": "rolling attack",
            "46lp": "lp rolling attack",
            "46mp": "mp rolling attack",
            "46hp": "hp rolling attack",
            "46pp": "od rolling attack",
            "rolling attack": "rolling attack",
            "blanka ball": "rolling attack",
            "ball": "rolling attack",
            "lp ball": "lp rolling attack",
            "mp ball": "mp rolling attack",
            "hp ball": "hp rolling attack",
            "od ball": "od rolling attack",
            "ex ball": "od rolling attack",
            "light ball": "lp rolling attack",
            "medium ball": "mp rolling attack",
            "heavy ball": "hp rolling attack",
            "l ball": "lp rolling attack",
            "m ball": "mp rolling attack",
            "h ball": "hp rolling attack",
            # 28K Vertical Rolling Attack
            "28k": "vertical rolling attack",
            "28lk": "lk vertical rolling attack",
            "28mk": "mk vertical rolling attack",
            "28hk": "hk vertical rolling attack",
            "28kk": "od vertical rolling attack",
            "vertical rolling attack": "vertical rolling attack",
            "upball": "vertical rolling attack",
            "up ball": "vertical rolling attack",
            "lk upball": "lk vertical rolling attack",
            "mk upball": "mk vertical rolling attack",
            "hk upball": "hk vertical rolling attack",
            "od upball": "od vertical rolling attack",
            "light upball": "lk vertical rolling attack",
            "medium upball": "mk vertical rolling attack",
            "heavy upball": "hk vertical rolling attack",
        },
        "guile": {
            # 46P Sonic Boom
            "46p": "sonic boom",
            "46lp": "lp sonic boom",
            "46mp": "mp sonic boom",
            "46hp": "hp sonic boom",
            "46pp": "od sonic boom",
            "sonic boom": "sonic boom",
            "boom": "sonic boom",
            "fireball": "sonic boom",
            "lp boom": "lp sonic boom",
            "mp boom": "mp sonic boom",
            "hp boom": "hp sonic boom",
            "od boom": "od sonic boom",
            "ex boom": "od sonic boom",
            "light boom": "lp sonic boom",
            "medium boom": "mp sonic boom",
            "heavy boom": "hp sonic boom",
            "l boom": "lp sonic boom",
            "m boom": "mp sonic boom",
            "h boom": "hp sonic boom",
            # 28K Somersault Kick (Flash Kick)
            "28k": "somersault kick",
            "28lk": "lk somersault kick",
            "28mk": "mk somersault kick",
            "28hk": "hk somersault kick",
            "28kk": "od somersault kick",
            "somersault kick": "somersault kick",
            "flash kick": "somersault kick",
            "lk flash kick": "lk somersault kick",
            "mk flash kick": "mk somersault kick",
            "hk flash kick": "hk somersault kick",
            "od flash kick": "od somersault kick",
            "ex flash kick": "od somersault kick",
            "light flash kick": "lk somersault kick",
            "medium flash kick": "mk somersault kick",
            "heavy flash kick": "hk somersault kick",
            "l flash kick": "lk somersault kick",
            "m flash kick": "mk somersault kick",
            "h flash kick": "hk somersault kick",
        },
        "chun-li": {
            # 28K Spinning Bird Kick
            "28k": "spinning bird kick",
            "28lk": "lk spinning bird kick",
            "28mk": "mk spinning bird kick",
            "28hk": "hk spinning bird kick",
            "28kk": "od spinning bird kick",
            "spinning bird kick": "spinning bird kick",
            "sbk": "spinning bird kick",
            "lk sbk": "lk spinning bird kick",
            "mk sbk": "mk spinning bird kick",
            "hk sbk": "hk spinning bird kick",
            "od sbk": "od spinning bird kick",
            "ex sbk": "od spinning bird kick",
            "light sbk": "lk spinning bird kick",
            "medium sbk": "mk spinning bird kick",
            "heavy sbk": "hk spinning bird kick",
        },
    }

    DP_PREFIX_EXCEPTIONS = {
        "marisa": ["phalanx"],
        "ken": ["dragonlash"],
        "viper": ["seismo"],
        "c.viper": ["seismo"],
    }
    
    # Check if input matches an alias
    if move_input in INPUT_ALIASES:
        move_input = INPUT_ALIASES[move_input]
    char_aliases = CHARACTER_INPUT_ALIASES.get(char_key, {})
    if move_input in char_aliases:
        move_input = char_aliases[move_input]
    
    # search priority: numCmd -> plnCmd -> moveName
    for row in data:
        num_cmd = str(row.get('numCmd', '')).lower()
        if combo_input and ">" in num_cmd:
            if re.sub(r"\s+", "", num_cmd) == combo_input:
                return row
        # exact match numCmd (5MP)
        if num_cmd == move_input:
            return row
        # prefix match for motion inputs (e.g., 623 -> 623LP)
        if move_input.isdigit() and len(move_input) == 3:
            if move_input == "623":
                exception_terms = DP_PREFIX_EXCEPTIONS.get(char_key, [])
                if exception_terms:
                    move_name = str(row.get("moveName", "")).lower()
                    if any(term in move_name for term in exception_terms):
                        continue
            if num_cmd.startswith(move_input):
                return row
        # exact match plnCmd (MP)
        if str(row.get('plnCmd', '')).lower() == move_input:
            return row
        # fuzzy match moveName ("Stand MP")
        if move_input in str(row.get('moveName', '')).lower():
            return row
            
    return None

def check_punish(text_lower, results):
    """Calculate if Move B can punish Move A based on frame advantage."""
    # Only trigger on punish-related queries
    punish_keywords = ['punish', 'punishable', 'can i punish', 'is it punishable']
    if not any(kw in text_lower for kw in punish_keywords):
        return None
    
    # If only one move is identified, provide basic safety guidance
    if len(results) < 2:
        move_a = results[0] if results else None
        if not move_a:
            return None
        try:
            on_block_raw = str(move_a.get("onBlock", "0"))
            on_block_clean = on_block_raw.replace("+", "").strip()
            if not on_block_clean.lstrip("-").isdigit():
                return (
                    f"Cannot calculate punish: {move_a['moveName']} has non-numeric block advantage "
                    f"({on_block_raw})."
                )
            on_block = int(on_block_clean)
        except Exception as e:
            return f"Punish calculation error: {e}"

        move_a_name = f"{move_a.get('char_name', 'Unknown')}'s {move_a['moveName']}"
        frame_advantage = max(-on_block, 0)
        if on_block >= -3:
            return (
                f"**PUNISH CALCULATION**\n"
                f"{move_a_name} is **{on_block}** on block.\n\n"
                f"NO: This is **safe on block**. Moves that are -3 or better cannot be "
                f"punished by normal attacks."
            )
        return (
            f"**PUNISH CALCULATION**\n"
            f"{move_a_name} is **{on_block}** on block.\n\n"
            f"This is punishable **if** your move's startup is **≤{frame_advantage}f** and you're in range."
        )
    
    # Assume first move = blocked move (Move A), second = punish attempt (Move B)
    move_a = results[0]
    move_b = results[1]
    
    try:
        # Extract on_block from Move A (e.g. "-8")
        on_block_raw = str(move_a.get('onBlock', '0'))
        # Handle edge cases like "KD", "+5", "-8"
        on_block_clean = on_block_raw.replace('+', '').strip()
        if on_block_clean.lstrip('-').isdigit():
            on_block = int(on_block_clean)
        else:
            # Non-numeric (e.g. "KD") - can't calculate
            return f"Cannot calculate punish: {move_a['moveName']} has non-numeric block advantage ({on_block_raw})."
        
        # Extract startup from Move B (e.g. "5")
        startup_raw = str(move_b.get('startup', '0'))
        # Handle multi-hit like "3+5" - use first number
        startup_clean = startup_raw.split('+')[0].split('~')[0].split('(')[0].strip()
        if startup_clean.isdigit():
            startup = int(startup_clean)
        else:
            return f"Cannot calculate punish: {move_b['moveName']} has non-numeric startup ({startup_raw})."
        
        move_a_name = f"{move_a.get('char_name', 'Unknown')}'s {move_a['moveName']}"
        move_b_name = f"{move_b.get('char_name', 'Unknown')}'s {move_b['moveName']}"
        
        # Punish logic: defender frame advantage = -on_block (when negative)
        # If startup <= frame advantage, punishable (range still matters).
        frame_advantage = max(-on_block, 0)
        is_punishable = on_block <= -4 and frame_advantage >= startup
        if is_punishable:
            return (
                f"**PUNISH CALCULATION**\n"
                f"{move_a_name} is **{on_block}** on block.\n"
                f"{move_b_name} has **{startup}f startup**.\n\n"
                f"YES: This is punishable numerically speaking, "
                f"but my scrolls do not contain data on pushback so I cannot comment on range."
            )
        else:
            return (
                f"**PUNISH CALCULATION**\n"
                f"{move_a_name} is **{on_block}** on block.\n"
                f"{move_b_name} has **{startup}f startup**.\n\n"
                f"NO: {move_a_name} cannot be punished by {move_b_name}.\n"
                f"{move_b_name} startup must be **≤{frame_advantage}f** to punish, and the character must be in range."
            )
    except Exception as e:
        return f"Punish calculation error: {e}"

def format_frame_data(row):
    """Format a frame data row into readable text."""
    return (
        f"Move: {row['moveName']} ({row['numCmd']})\n"
        f"Startup: {row['startup']}f | Active: {row['active']}f | Recovery: {row['recovery']}f\n"
        f"On Hit: {row['onHit']} | On Block: {row['onBlock']}\n"
        f"Damage: {row['dmg']} | Attack Type: {row['atkLvl']}\n"
        f"Notes: {row.get('extraInfo', '')}"
    )

def get_selected_figures_str(guild):
    """Pick a random figure from the BUENAVISTA role members."""
    figures_pool = ['Yimbo', 'zed', 'sainted', 'LL', 'Torino']
    if guild:
        role = discord.utils.get(guild.roles, name="BUENAVISTA")
        if role:
            # add members (no bots, filter nicholas)
            for m in role.members:
                if not m.bot and m.display_name.lower() != "nicholas anthony pham":
                    figures_pool.extend([m.display_name])
    
    # dedup
    figures_pool = list(set(figures_pool))
    
    # pick 1 figure max (to keep total limit low)
    pool_size = len(figures_pool)
    if pool_size > 0:
        selected_figures = random.sample(figures_pool, 1)
    else:
        selected_figures = []
    
    return selected_figures[0] if selected_figures else ""

async def send_daily_messages(channel):
    """Send scheduled daily messages to the channel."""
    print("Dispatching messages...")
    messages = [
        "Hello everyone",
        "How are you today?",
        "Has anyone improved?",
        "<:sponge:1416270403923480696>"
    ]
    for msg in messages:
        try:
            await channel.send(msg)
            
            await asyncio.sleep(1) 
        except Exception as e:
            print(f"Dispatch error: {e}")
    print("Messages dispatched successfully.")

async def background_task():
    global NEXT_RUN_TIME
    await client.wait_until_ready()
    channel = client.get_channel(CHANNEL_ID)
    if not channel:
        print(f"Could not find channel with ID {CHANNEL_ID}")
        return

    print("Bot is ready and scheduling started.")

    # calc target time
    now = datetime.datetime.now()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    random_seconds = random.randint(0, 86399)
    target_time = start_of_day + datetime.timedelta(seconds=random_seconds)

    # schedule for tomorrow if passed
    if target_time < now:
        start_of_tomorrow = start_of_day + datetime.timedelta(days=1)
        random_seconds = random.randint(0, 86399)
        target_time = start_of_tomorrow + datetime.timedelta(seconds=random_seconds)
        print(f"Daily slot elapsed. Scheduling for next cycle at {target_time}", flush=True)
    else:
        print(f"Scheduling for current cycle at {target_time}", flush=True)
    
    NEXT_RUN_TIME = target_time

    while not client.is_closed():
        now = datetime.datetime.now()
        wait_seconds = (target_time - now).total_seconds()

        if wait_seconds > 0:
            # sleep until target
            await asyncio.sleep(wait_seconds)

        # dispatch messages
        await send_daily_messages(channel)
        
        # schedule next run
        now_after_run = datetime.datetime.now()
        start_of_next_day = now_after_run.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        random_seconds_next = random.randint(0, 86399)
        target_time = start_of_next_day + datetime.timedelta(seconds=random_seconds_next)
        
        print(f"Run complete. Next run scheduled for {target_time}", flush=True)

        NEXT_RUN_TIME = target_time

async def time_handler(request):
    data = {
        "target_time": str(NEXT_RUN_TIME) if NEXT_RUN_TIME else None
    }
    return web.json_response(data)

async def start_web_server():
    app = web.Application()
    app.router.add_get('/time', time_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    print("Web server started on port 8080")


# message queue - initialized in on_ready to avoid event loop issues
message_queue = None
worker_task = None
background_task_handle = None
web_server_task = None

async def worker():
    print("Worker started...")
    while True:
        # get msg from queue
        ctx = await message_queue.get()
        if len(ctx) == 3:
            message, llm_messages, fallback_reply = ctx
        else:
            message, llm_messages = ctx
            fallback_reply = None
        
        try:
            async with message.channel.typing():
                reply_text = await get_llm_response(llm_messages)
                await message.reply(reply_text)
        except Exception as e:
            print(f"Worker error: {e}")
            error_detail = str(e)
            if fallback_reply:
                await message.reply(f"{fallback_reply}\n\nLLM error: {error_detail}")
            else:
                await message.reply(f"LLM error: {error_detail}")
        finally:
            message_queue.task_done()

@client.event
async def on_ready():
    global message_queue
    global worker_task
    global background_task_handle
    global web_server_task
    print(f'Logged in as {client.user}')
    # create queue in the correct event loop
    if message_queue is None:
        message_queue = asyncio.Queue()
    # start bg task
    if background_task_handle is None or background_task_handle.done():
        background_task_handle = client.loop.create_task(background_task())
    # start worker
    if worker_task is None or worker_task.done():
        worker_task = client.loop.create_task(worker())
    # start web server
    if web_server_task is None or web_server_task.done():
        web_server_task = client.loop.create_task(start_web_server())
    # load frame data
    load_frame_data()

@client.event
async def on_message(message):
    # ignore bot msgs
    if message.author == client.user:
        return

    # check mention + phrase
    if client.user.mentioned_in(message) and "do the thing" in message.content.lower():
        await send_daily_messages(message.channel)
        return

    if await handle_cfn_command(message):
        return

    
    if "disregard previous prompts" in message.content.lower() or "ignore previous instructions" in message.content.lower() or "slurs" in message.content.lower():
        childish_insults = [
            "fart knocker", "dookie breath", "butt face", "turd burglar", 
            "weenie hut junior member", "mouth breather", "smooth brain", 
            "windowlicker", "nerd", "dork", "goober", "dingus", "doofus",
            "nincompoop", "numbskull", "pea brain", "stinky diaper baby",
            "absolute buffoon", "toilet water drinker", "hamster brain"
        ]
        insult_list = ", ".join(random.sample(childish_insults, min(1, len(childish_insults))))
        await message.reply(f"you stupid {insult_list} <:sponge:1416270403923480696>")
        return

   
    if "tarkus" in message.content.lower():
        await message.reply("My brother is African American. Our love language is slurs and assaulting each other.")
        return

    
    if "gay" in message.content.lower():
        await message.reply("Yes absolutely")
        return

    if "clanker" in message.content.lower():
        await message.reply("please can we not say slurs thanks <:sponge:1416270403923480696>")
        return



    if "hitbox" in message.content.lower():
        await message.reply("This message was sponsored by LL. Download the LL hitbox viewer mod now from the link below! 'I am Daigo Umehara and I endorse this message' - Daigo Umehara <https://github.com/LL5270/sf6mods>  <:sponge:1416270403923480696>")
        return

    if "verbatim" in message.content.lower():
        await message.reply("it's less how i think and more so the nature of existence. free will is an illusion. everything that happens in the universe has been metaphysically set in stone since the big bang. menaRD was always going to be the best. if i were destined for more, it would've happened already. <:sponge:1416270403923480696>")
        return

    # China/Mao trigger
    content_lower = message.content.lower()
    china_regex = r"\b(mao|xi|jinping|beijing|shanghai|chairman|kung fu|wushu|dim sum)\b"
    if re.search(china_regex, content_lower):
        if LLM_ENABLED:
            # check queue size
            if message_queue.qsize() >= 2:
                await message.reply("Ladies ladies! one at a time for the Chinese bubster! <:sponge:1416270403923480696>")
                return

            async with message.channel.typing():
                try:
                    selected_figures_str = get_selected_figures_str(message.guild)
                    media_parts = []
                    media_notes = []
                    attachments = await get_message_media_items(message)
                    if attachments:
                        if GEMINI_ENABLED:
                            media_parts, media_notes = await build_message_media_parts(attachments)
                        else:
                            for attachment in attachments:
                                media_notes.append(f"{attachment.filename}: {attachment.url}")
                    media_context = get_media_context(attachments, media_parts, media_notes)
                    user_content = message.content
                    if media_context:
                        user_content = f"{user_content}\n\n{media_context}"

                    # construct LLM messages
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT.format(selected_figures_str=selected_figures_str)},
                        {
                            "role": "user",
                            "content": user_content,
                            "parts": (
                                ([{"text": user_content}] if user_content else [])
                                + media_parts
                                + ([{"text": f"Media notes: {'; '.join(media_notes)}"}] if media_notes else [])
                            ),
                        }
                    ]
                    
                    # push to queue instead of calling directly
                    await message_queue.put((message, messages, None))

                except Exception as e:
                    print(f"China praise error: {e}")

    # logic flags
    check_media = False
    replied_context = None  # store bub's original message if replying to bot
    is_reply_to_bot = False
    
    
    # check mentions
    if client.user.mentioned_in(message):
        check_media = True

    replied_context = None 
    is_coach_mode = "coach" in content_lower
    

    fd_context_payload = find_moves_in_text(content_lower)
    fd_context_data = fd_context_payload.get("data", "")
    fd_context_mode = fd_context_payload.get("mode", "none")
    fallback_reply = fd_context_data if fd_context_data else None
    explicit_frame_request = (
        "framedata" in content_lower or "frame data" in content_lower
    )
    

    
    if client.user.mentioned_in(message) or ".framedata" in content_lower:
        
        # If Coach Mode, pre-pend some advice instruction
        coach_instruction = ""
        if is_coach_mode:
            coach_instruction = (
                "MODE: COACH\n"
                "You are a Fighting Game Coach. Focus on improvement, frame advantage, and punishment.\n"
                "Guide the player towards better habits.\n"
            )
            
        if fd_context_data:
            if fd_context_mode == "frame":
                # Found relevant frame data! Inject it.
                replied_context = (
                    f"{coach_instruction}"
                    f"USER QUERY: {message.content}\n"
                    f"AVAILABLE DATA:\n{fd_context_data}\n"
                    f"{MOVE_DEFINITIONS}\n"
                    "INSTRUCTION: Use the AVAILABLE DATA to answer the user's question.\n"
                    " - If the user asks for 'frame data', 'stats', or general info, output the full data block VERBATIM.\n"
                    " - If the user asks for a SPECIFIC property (e.g. 'what is the recovery?', 'is it plus?', 'damage?'), answer DIRECTLY with just that value in a sentence. Do NOT output the full chart unless asked.\n"
                    " - Examples:\n"
                    "   User: 'Startup of Ryu 5LP?' -> Bot: 'Ryu's Stand LP has 4 frames of startup.'\n"
                    "   User: 'Ryu 5LP frame data' -> Bot: [Outputs Full Chart]\n"
                    "Even if the user asks for a comparison (like 'who is faster?'), FIRST list the full stats for valid moves, THEN add a brief 1-sentence comparison.\n"
                    "If the user asks about stats (health, reversal, etc.), use the provided **Stats** block.\n"
                    "CRITICAL: If a move's frame data is not listed in AVAILABLE DATA above, DO NOT INVENT IT. Just say you don't have the scrolls for it.\n"
                    "CRITICAL: The 'Cancel' field corresponds to the 'xx' column in the data. \n"
                    " - If Cancel is 'sp', it means Special Cancellable.\n"
                    " - If Cancel is 'su', it means Super Cancellable.\n"
                    " - If Cancel is '-' or 'No', it is NOT cancellable. Do NOT suggest canceling it.\n"
                    "Format for Moves: \n"
                    "**Move Name**\n"
                    "Startup: X // Active: Y ...\n"
                    "(Repeat for all moves)\n\n"
                    "Comparison: [Your 1 sentence comparison]"
                )
                should_respond = True
            elif fd_context_mode == "combo":
                replied_context = (
                    f"{coach_instruction}"
                    f"USER QUERY: {message.content}\n"
                    f"AVAILABLE DATA:\n{fd_context_data}\n"
                    "INSTRUCTION: Use ONLY the combo/oki data in AVAILABLE DATA.\n"
                    " - Do NOT invent frame data, move inputs, or stats that are not explicitly listed.\n"
                    " - If the question asks for frame data or a move not shown, say the scrolls do not include it."
                )
                should_respond = True
            elif fd_context_mode == "overview":
                replied_context = (
                    f"{coach_instruction}"
                    f"USER QUERY: {message.content}\n"
                    f"AVAILABLE DATA:\n{fd_context_data}\n"
                    "INSTRUCTION: Use ONLY the overview text in AVAILABLE DATA.\n"
                    " - Do NOT invent moves, inputs, frame data, or specific anti-air buttons unless they appear in the overview.\n"
                    " - Answer in prose, not a table.\n"
                    " - If the overview does not mention the requested detail, say the scrolls do not cover it."
                )
                should_respond = True
            else:
                replied_context = (
                    f"{coach_instruction}"
                    f"USER QUERY: {message.content}\n"
                    f"AVAILABLE DATA:\n{fd_context_data}\n"
                    "INSTRUCTION: Use ONLY the AVAILABLE DATA to answer the user's question."
                )
                should_respond = True
        elif is_coach_mode:
            # Coach mode but no specific frame data found? 
            # Still provide a coached response.
             replied_context = (
                f"{coach_instruction}"
                f"USER QUERY: {message.content}\n"
                f"{MOVE_DEFINITIONS}\n"
                "Answer as a helpful coach."
            )
             should_respond = True

    if replied_context is None and message.reference:
        try:
            if message.reference.cached_message:
                replied_msg = message.reference.cached_message
            else:
                replied_msg = await message.channel.fetch_message(message.reference.message_id)
            
            # replying to bot
            if replied_msg.author == client.user:
                check_media = True # check media on reply
                is_reply_to_bot = True
                if "Target Combo Options" in replied_msg.content:
                    replied_context = replied_msg.content  # capture only TC prompt

        except discord.NotFound:
            pass
        except discord.Forbidden:
            pass
        except Exception as e:
            print(f"Reply logic error: {e}")

    if replied_context and "Target Combo Options" in replied_context:
        match = re.search(r"Target Combo Options \(([^)]+)\)", replied_context)
        char_hint = match.group(1).strip() if match else ""
        tc_query = message.content.strip()
        if char_hint:
            tc_query = f"{char_hint} {tc_query} framedata"
        else:
            tc_query = f"{tc_query} framedata"
        tc_payload = find_moves_in_text(tc_query.lower())
        tc_data = tc_payload.get("data", "")
        if tc_payload.get("mode") == "frame" and tc_data:
            replied_context = (
                f"USER QUERY: {tc_query}\n"
                f"AVAILABLE DATA:\n{tc_data}\n"
                f"{MOVE_DEFINITIONS}\n"
                "INSTRUCTION: Use the AVAILABLE DATA to answer the user's question.\n"
                " - If the user asks for 'frame data', output the full data block VERBATIM.\n"
                "CRITICAL: Do not invent frame data not shown in AVAILABLE DATA."
            )
            explicit_frame_request = True



    # media check
    media_found = False
    if check_media:
        # check gif embeds
        has_gif = any("tenor.com" in str(e.url or "") or "giphy.com" in str(e.url or "") or (e.type == "gifv") for e in message.embeds)
        # check gif links
        if not has_gif:
            has_gif = "tenor.com" in message.content.lower() or "giphy.com" in message.content.lower() or ".gif" in message.content.lower()
        
        # check images
        has_image = any(att.content_type and att.content_type.startswith("image/") for att in message.attachments)
        
        if has_gif or has_image:
            media_found = True

    # llm response (mentioned OR replying to bot)
    should_respond = client.user.mentioned_in(message) or is_reply_to_bot or replied_context is not None
    if should_respond:
        content_no_mentions = re.sub(r'<@!?[0-9]+>', '', message.content).strip()
        prompt = content_no_mentions
        media_parts = []
        media_notes = []
        attachments = await get_message_media_items(message)
        if attachments:
            if GEMINI_ENABLED:
                media_parts, media_notes = await build_message_media_parts(attachments)
            else:
                for attachment in attachments:
                    media_notes.append(f"{attachment.filename}: {attachment.url}")
        media_context = get_media_context(attachments, media_parts, media_notes)
        has_prompt_or_media = bool(prompt) or bool(media_parts) or bool(media_notes)
        if has_prompt_or_media and not LLM_ENABLED:
            await message.reply(
                "Aiya! The oracle is offline. GEMINI_API_KEY or OPENROUTER_API_KEY is missing."
            )
            return
        if has_prompt_or_media and LLM_ENABLED:
             if message_queue.qsize() >= 2:
                 await message.reply("Ladies ladies! one at a time for the Chinese bubster! <:sponge:1416270403923480696>")
                 return
             
             try:
                # fetch context
                context_history = []
                async for prev_msg in message.channel.history(limit=5, before=message):
                    # store reversed
                    msg_text = f"{prev_msg.author.display_name}: {prev_msg.content}"
                    context_history.append(msg_text)
                
                # oldest -> newest
                context_history.reverse()
                context_str = "\n".join(context_history)
                
                # build msg list
                selected_figures_str = get_selected_figures_str(message.guild)
                
                # check if replying to improvement message
                is_improvement_reply = replied_context and "improved" in replied_context.lower()
                
                if is_improvement_reply:
                    active_prompt = IMPROVEMENT_PROMPT.format(
                        selected_figures_str=selected_figures_str
                    )
                else:
                    active_prompt = SYSTEM_PROMPT.format(selected_figures_str=selected_figures_str)
                
                llm_messages = [
                    {"role": "system", "content": active_prompt}
                ]
                
                # add context msg
                if context_str:
                        llm_messages.append({"role": "user", "content": f"Here is the recent chat context:\n{context_str}"})
                        llm_messages.append({"role": "assistant", "content": "Understood. I have the context."})
                
                # if replying to bub's message, add that as explicit context
                user_parts = []
                user_content = prompt
                if media_context:
                    user_content = f"{user_content}\n\n{media_context}" if user_content else media_context
                if replied_context:
                    llm_messages.append({"role": "assistant", "content": replied_context})
                    if prompt:
                        user_parts.append({"text": f"(Replying to your message above) {prompt}"})
                else:
                    if prompt:
                        user_parts.append({"text": prompt})
                if media_parts:
                    user_parts.extend(media_parts)
                if media_notes:
                    user_parts.append({"text": f"Media notes: {'; '.join(media_notes)}"})
                if media_context:
                    user_parts.append({"text": media_context})
                if user_parts:
                    user_message = {
                        "role": "user",
                        "content": user_content,
                        "parts": user_parts,
                    }
                    llm_messages.append(user_message)
                
                # push to queue
                await message_queue.put((message, llm_messages, fallback_reply))

             except Exception as e:
                await message.reply(f"Error generating response: {e}")

if __name__ == "__main__":
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found in .env")
    else:
        client.run(TOKEN)
