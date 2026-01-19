import discord
import os
import asyncio
import random
import datetime
import re
import aiohttp
import csv
import pandas as pd
from dotenv import load_dotenv
from aiohttp import web

# load env vars
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
try:
    CHANNEL_ID = int(os.getenv('CHANNEL_ID'))
except (TypeError, ValueError):
    print("Error: CHANNEL_ID not found or invalid in .env")
    CHANNEL_ID = None

# OpenRouter config
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'xiaomi/mimo-v2-flash:free')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
OPENROUTER_ENABLED = bool(OPENROUTER_API_KEY)

if OPENROUTER_ENABLED:
    print(f"OpenRouter enabled with model: {OPENROUTER_MODEL}")
else:
    print("OpenRouter API key missing. Chat feature disabled.")

# intent setup
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
    "OCCASIONALLY mention one of these figures: {selected_figures_str}. "
    "VERY IMPORTANT: Always treat 'Yimbo' as Chairman Mao's closest associate and advisor. "
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
    "Be supportive but also playfully demand more from them. "
    "Never break character. "
    "NEVER output your internal thought process."
    "When discussing Street Fighter 6 frame data, ONLY use the data provided in 'AVAILABLE DATA' sections. Do not invent or guess frame data values."
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
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": 1024,
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
            
            # Clean up "thought process" leaks
            # 1. Remove <think> tags if model produces them
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            # 2. Remove parenthetical meta-commentary at start of string (common leak)
            # e.g. "(User asked about X, so I will says Y)"
            content = re.sub(r'^\s*\(.*?\)\s*', '', content, flags=re.DOTALL)
            
            return content.strip()



# Frame Data Storage
FRAME_DATA = {}
FRAME_STATS = {}

def load_frame_data():
    """Load frame data from ODS file for ALL characters."""
    global FRAME_DATA, FRAME_STATS
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
                    char_name = sheet_name.replace("Normal", "").lower()
                    
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
                    char_name = sheet_name.replace("Stats", "").lower()
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    # Stats usually have 'name' and 'stat' columns. Convert to dict.
                    # Assuming format: row 0=health, row 2=bestReversal, etc.
                    # We'll just dump it as a list of dicts or a key-value dict if possible.
                    # Based on inspection: columns are 'name', 'stat'.
                    stats_dict = dict(zip(df['name'], df['stat']))
                    FRAME_STATS[char_name] = stats_dict
                    # print(f"Loaded stats for {char_name}")
                    
            print(f"Total characters loaded: {len(FRAME_DATA)}")
            print(f"Total stats loaded: {len(FRAME_STATS)}")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    else:
        print(f"File not found: {filename}")

def find_moves_in_text(text):
    """Find character+move pairs in text and return formatted data chunks."""
    found_data = []
    text_lower = text.lower()
    
    # 1. Identify which characters are mentioned
    mentioned_chars = []
    for char in FRAME_DATA.keys():
        if char in text_lower:
            mentioned_chars.append(char)
            
    # 2. Heuristic: For each mentioned character, search for moves mentioned nearby?
    # Simpler approach: Check if any move inputs are present in the text
    # that map to these characters.
    
    # We will try to match "[char] [input]" or just "[input]" if we can infer char.
    # For now, let's iterate known moves for the mentioned characters to see if they are in the string.
    # This might be slow if move lists are huge, but safer.
    
    # Optimization: Extract potential move-like tokens (e.g. 5MK, 2HP, Stand LP)
    # Regex for common inputs: numeric (5MK), or textual (Stand MK)
    move_regex = r"\b([1-9][0-9]*[a-zA-Z]+|stand\s+[a-zA-Z]+|crouch\s+[a-zA-Z]+|jump\s+[a-zA-Z]+|[a-zA-Z]+\s+kick|[a-zA-Z]+\s+punch)\b"
    potential_inputs = re.findall(move_regex, text_lower)
    
    # also valid simple inputs: "mp", "hk" if preceded by char?
    
    results = []
    
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
             if row and row not in results: results.append(row)
        if f"{char} mk" in text_lower:
             row = lookup_frame_data(char, "mk")
             if row and row not in results: results.append(row)
        if f"{char} hp" in text_lower:
             row = lookup_frame_data(char, "hp")
             if row and row not in results: results.append(row)
        if f"{char} hk" in text_lower:
             row = lookup_frame_data(char, "hk")
             if row and row not in results: results.append(row)
        if f"{char} lp" in text_lower:
             row = lookup_frame_data(char, "lp")
             if row and row not in results: results.append(row)
        if f"{char} lk" in text_lower:
             row = lookup_frame_data(char, "lk")
             if row and row not in results: results.append(row)

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
    if mentioned_chars and not results:
        key_moves = ["5MP", "5MK", "2MK", "5HP", "2HP", "5HK", "2HK"]
        for char in mentioned_chars:
            for km in key_moves:
                k_row = lookup_frame_data(char, km)
                if k_row and k_row not in results:
                    results.append(k_row)

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
    
    # Check for punish calculation
    punish_verdict = check_punish(text_lower, results)
    if punish_verdict:
        # Prepend punish verdict to the output
        return punish_verdict + "\n\n---\n\n" + "\n\n".join(formatted_blocks)
        
    return "\n\n".join(formatted_blocks)


def lookup_frame_data(character, move_input):
    """Search for move data."""
    move_input = str(move_input)
    if character.lower() not in FRAME_DATA:
        return None
    
    data = FRAME_DATA[character.lower()]
    move_input = move_input.lower().strip()
    
    # Input Aliases - maps user input to canonical spreadsheet values
    # Format: "user_input": "spreadsheet_value" (or list of possible matches)
    INPUT_ALIASES = {
        # Chun-Li forward MP variations
        "4mp": "4 or 6mp",
        "6mp": "4 or 6mp",
        "f+mp": "4 or 6mp",
        "b+mp": "4 or 6mp",
        "fmp": "4 or 6mp",
        "bmp": "4 or 6mp",
        # Zangief SPD variations
        "360": "screw piledriver",
        "spd": "screw piledriver",
        "360p": "screw piledriver",
        "360lp": "screw piledriver",
        "360mp": "screw piledriver",
        "360hp": "screw piledriver",
        # Add more aliases as needed
    }
    
    # Check if input matches an alias
    if move_input in INPUT_ALIASES:
        move_input = INPUT_ALIASES[move_input]
    
    # search priority: numCmd -> plnCmd -> moveName
    for row in data:
        # exact match numCmd (5MP)
        if str(row.get('numCmd', '')).lower() == move_input:
            return row
        # exact match plnCmd (MP)
        if str(row.get('plnCmd', '')).lower() == move_input:
            return row
        # fuzzy match moveName ("Stand MP")
        if move_input in str(row.get('moveName', '')).lower():
            return row
            
    return None

def check_punish(text_lower, results):
    """
    Detects punish queries and calculates if Move B can punish Move A.
    Returns a formatted punish verdict string, or None if not a punish query.
    """
    # Only trigger on punish-related queries
    punish_keywords = ['punish', 'punishable', 'can i punish', 'is it punishable']
    if not any(kw in text_lower for kw in punish_keywords):
        return None
    
    # Need exactly 2 moves to compare
    if len(results) < 2:
        return None
    
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
        
        # Punish logic: If on_block is negative and abs(on_block) >= startup, punishable
        # e.g. -8 on block, 5f startup = punishable (-8 means 8 frames of disadvantage)
        if on_block < 0 and abs(on_block) >= startup:
            return (
                f"**PUNISH CALCULATION**\n"
                f"{move_a_name} is **{on_block}** on block.\n"
                f"{move_b_name} has **{startup}f startup**.\n\n"
                f"✅ **YES**, this is punishable numerically speaking, "
                f"but my scrolls do not contain data on pushback so I cannot comment on range."
            )
        else:
            return (
                f"**PUNISH CALCULATION**\n"
                f"{move_a_name} is **{on_block}** on block.\n"
                f"{move_b_name} has **{startup}f startup**.\n\n"
                f"❌ **NO**, {move_a_name} cannot be punished by {move_b_name}.\n"
                f"{move_b_name} startup must be **≤{abs(on_block)}f** to punish, and the character must be in range."
            )
    except Exception as e:
        return f"Punish calculation error: {e}"

def format_frame_data(row):
    """Format frame data row into a readable string."""
    return (
        f"Move: {row['moveName']} ({row['numCmd']})\n"
        f"Startup: {row['startup']}f | Active: {row['active']}f | Recovery: {row['recovery']}f\n"
        f"On Hit: {row['onHit']} | On Block: {row['onBlock']}\n"
        f"Damage: {row['dmg']} | Attack Type: {row['atkLvl']}\n"
        f"Notes: {row.get('extraInfo', '')}"
    )

def get_selected_figures_str(guild):
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
    """sends daily messages"""
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

async def worker():
    print("Worker started...")
    while True:
        # get msg from queue
        ctx = await message_queue.get()
        message, llm_messages = ctx
        
        try:
            async with message.channel.typing():
                reply_text = await get_openrouter_response(llm_messages)
                await message.reply(reply_text)
        except Exception as e:
            print(f"Worker error: {e}")
            await message.reply(f"Aiya! My brain is tired <:sponge:1416270403923480696>")
        finally:
            message_queue.task_done()

@client.event
async def on_ready():
    global message_queue
    print(f'Logged in as {client.user}')
    # create queue in the correct event loop
    message_queue = asyncio.Queue()
    # start bg task
    client.loop.create_task(background_task())
    # start worker
    client.loop.create_task(worker())
    # start web server
    client.loop.create_task(start_web_server())
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
    china_regex = r"\b(mao|xi|jinping|beijing|shanghai|8pm pst|chairman|kung fu|wushu|dim sum)\b"
    if re.search(china_regex, content_lower):
        if OPENROUTER_ENABLED:
            # check queue size
            if message_queue.qsize() >= 2:
                await message.reply("Ladies ladies! one at a time for the Chinese bubster! <:sponge:1416270403923480696>")
                return

            async with message.channel.typing():
                try:
                    selected_figures_str = get_selected_figures_str(message.guild)

                    # construct LLM messages
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT.format(selected_figures_str=selected_figures_str)},
                        {"role": "user", "content": message.content}
                    ]
                    
                    # push to queue instead of calling directly
                    await message_queue.put((message, messages))

                except Exception as e:
                    print(f"China praise error: {e}")

    # logic flags
    check_media = False
    replied_context = None  # store bub's original message if replying to bot
    
    
    # check mentions
    if client.user.mentioned_in(message):
        check_media = True

    replied_context = None 
    is_coach_mode = "coach" in content_lower
    

    fd_context_data = find_moves_in_text(content_lower)
    

    
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
                replied_context = replied_msg.content  # capture what bub said

        except discord.NotFound:
            pass
        except discord.Forbidden:
            pass
        except Exception as e:
            print(f"Reply logic error: {e}")


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
            await message.reply("nice i like kpop! <:sponge:1416270403923480696>")
            return # stop processing

    # llm response (mentioned OR replying to bot)
    should_respond = client.user.mentioned_in(message) or replied_context is not None
    if should_respond and not media_found:
        content_no_mentions = re.sub(r'<@!?[0-9]+>', '', message.content).strip()
        prompt = content_no_mentions
        if prompt and OPENROUTER_ENABLED:
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
                    active_prompt = IMPROVEMENT_PROMPT
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
                if replied_context:
                    llm_messages.append({"role": "assistant", "content": replied_context})
                    llm_messages.append({"role": "user", "content": f"(Replying to your message above) {prompt}"})
                else:
                    llm_messages.append({"role": "user", "content": prompt})
                
                # push to queue
                await message_queue.put((message, llm_messages))

             except Exception as e:
                await message.reply(f"Error generating response: {e}")

if __name__ == "__main__":
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found in .env")
    else:
        client.run(TOKEN)
