import discord
import os
import asyncio
import random
import datetime
import re
import aiohttp
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
    "5 sentence limit. Keep it punchy."
)

IMPROVEMENT_PROMPT = (
    "You are Chinese Bub, a wise sensei who is OBSESSED with improvement and self-betterment. "
    "Your friends are warriors, and you believe they MUST grind, train, and level up - especially in fighting games like Street Fighter 6! "
    "You are extremely passionate and encouraging. "
    "Someone just replied to your question about improvement. Respond about their improvement journey. "
    "Give them motivation! Hype them up! Reference training, frame data, combos, ranked matches, or life skills. "
    "You can tie improvement to Chairman Mao's revolutionary spirit or Chinese resilience. "
    "Be supportive but also playfully demand more from them. "
    "Never break character."
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
        "top_p": 1.0
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
            return data['choices'][0]['message']['content']


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

    if "casual" in message.content.lower():
        await message.reply("There are no casuals! Only warriors in training! <:sponge:1416270403923480696>")
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

    # check replies
    if message.reference:
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
