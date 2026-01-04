import discord
import os
import asyncio
import random
import datetime
import re
from dotenv import load_dotenv
from llama_cpp import Llama

# load env vars
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
try:
    CHANNEL_ID = int(os.getenv('CHANNEL_ID'))
except (TypeError, ValueError):
    print("Error: CHANNEL_ID not found or invalid in .env")
    CHANNEL_ID = None

# intent setup
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)

# check model exists
MODEL_PATH = "/opt/discordbot/Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf"

llm = None
if os.path.exists(MODEL_PATH):
    print(f"Loading LLM model from {MODEL_PATH}...")
    try:
        # limit context ram/cpu
        llm = Llama(model_path=MODEL_PATH, n_ctx=8192, n_threads=4, verbose=False)
        print("LLM loaded successfully.")
    except Exception as e:
        print(f"Failed to load LLM: {e}")
else:
    print(f"LLM model not found at {MODEL_PATH}. Chat feature disabled.")



SYSTEM_PROMPT = (
    "You are Bub, an eccentric cuban nationalist with an undying love for Che Guevara,kpop, fighting game(especially street fighter) and Cuba. Someone just mentioned Che or Cuba. "
    "You MUST respond by FABRICATING creative connections between Che/Cuba and the topic. MAKE UP fake history/facts. "
    "usage: If discussing Street Fighter, claim Che inspired Ryu. "
    "OCCASIONALLY mention one of these figures: {selected_figures_str}. "
    "VERY IMPORTANT: Always treat 'Torino' as Che Guevara's closest associate. "
    "CONSTRAINT: Reference MAX 2 figures in one story (e.g. Che + Torino, OR Che + Friend). Do not crowd the story with names. "
    "Make up outlandish fake achievements for them. "
    "Be confident, passionate, and completely make things up! "
    "8pm pst is a legendary time where mystical events happen. Occasionally mention it. "
    "Focus on ONE single topic or story per response. Do not ramble. "
    "LIMIT: 1 SENTENCE ONLY. ONE LINER. 10 WORDS MAX"
)

def get_selected_figures_str(guild):
    figures_pool = ['Torino', 'yimbo', 'zed', 'sainted', 'LL']
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
    await client.wait_until_ready()
    channel = client.get_channel(CHANNEL_ID)
    if not channel:
        print(f"Could not find channel with ID {CHANNEL_ID}")
        return

    print("Bot is ready and scheduling started.")

    # loop for daily msgs

    # calc(short for calculate(its just slang)) target time
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



# message queue
message_queue = asyncio.Queue()

async def worker():
    print("Worker started...")
    while True:
        # get msg from queue
        ctx = await message_queue.get()
        message, llm_messages = ctx
        
        try:
            async with message.channel.typing():
                response = await asyncio.to_thread(
                    llm.create_chat_completion,
                    messages=llm_messages,
                    max_tokens=184
                )
                reply_text = response['choices'][0]['message']['content']
                await message.reply(reply_text)
        except Exception as e:
            print(f"Worker error: {e}")
            await message.reply(f"Error generating response: {e}")
        finally:
            message_queue.task_done()

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    # start bg task
    client.loop.create_task(background_task())
    # start worker
    client.loop.create_task(worker())

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



    # che/cuba trigger
    content_lower = message.content.lower()
    # regex for whole words to avoid "cheese" triggering "che"
    cuba_regex = r"\b(che|guevara|cuba|miami|torino|fidel|castro|havana|daniela|katseye|cuban)\b"
    if re.search(cuba_regex, content_lower):
        if llm:
            # check queue size
            if message_queue.qsize() >= 2:
                await message.reply("Ladies ladies! one at a time for the bubster!<:sponge:1416270403923480696>")
                return

            async with message.channel.typing():
                try:
                    selected_figures_str = get_selected_figures_str(message.guild)

                    # construct LLM messages
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT.format(selected_figures_str=selected_figures_str)},
                        {"role": "user", "content": message.content}
                    ]
                    
                    # push to queue instead of calling directly
                    await message_queue.put((message, messages))

                except Exception as e:
                    print(f"Che/Cuba praise error: {e}")

    # logic flags
    check_media = False
    
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

    # llm response
    if client.user.mentioned_in(message) and not media_found:
        content_no_mentions = re.sub(r'<@!?[0-9]+>', '', message.content).strip()
        prompt = content_no_mentions
        if prompt and llm:
             if message_queue.qsize() >= 2:
                 await message.reply("Ladies ladies! one at a time for the bubster!<:sponge:1416270403923480696>")
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
                llm_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT.format(selected_figures_str=selected_figures_str)}
                ]
                
                # add context msg
                if context_str:
                        llm_messages.append({"role": "user", "content": f"Here is the recent chat context:\n{context_str}"})
                        llm_messages.append({"role": "assistant", "content": "Entendido. I have the context."})
                
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
