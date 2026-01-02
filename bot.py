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
MODEL_PATH = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
if not os.path.exists(MODEL_PATH):
    # fallback to absolute path
    MODEL_PATH = "/opt/discordbot/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

llm = None
if os.path.exists(MODEL_PATH):
    print(f"Loading LLM model from {MODEL_PATH}...")
    try:
        # limit context ram/cpu
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)
        print("LLM loaded successfully.")
    except Exception as e:
        print(f"Failed to load LLM: {e}")
else:
    print(f"LLM model not found at {MODEL_PATH}. Chat feature disabled.")


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


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    # start bg task
    client.loop.create_task(background_task())

@client.event
async def on_message(message):
    # ignore bot msgs
    if message.author == client.user:
        return

    # check mention + phrase
    if client.user.mentioned_in(message) and "do the thing" in message.content.lower():
        await send_daily_messages(message.channel)









    
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

   
    if "tarkus" in message.content.lower():
        await message.reply("My brother is African American. Our love language is slurs and assaulting each other.")

    
    if "casual" in message.content.lower():
        await message.reply("Yes absolutely")

    # che/cuba trigger
    content_lower = message.content.lower()
    cuba_triggers = ["che guevara", "che ", " che", "cuba", "miami", "torino", "fidel castro", "havana", "daniela", "katseye"]
    if any(trigger in content_lower for trigger in cuba_triggers):
        if llm:
            async with message.channel.typing():
                try:
                    # get role members
                    figures_pool = ['Torino', 'yimbo', 'zed', 'sainted', 'LL']
                    if message.guild:
                        role = discord.utils.get(message.guild.roles, name="BUENAVISTA")
                        if role:
                            # add members (no bots)
                            figures_pool.extend([m.display_name for m in role.members if not m.bot])
                    
                    # dedup
                    figures_pool = list(set(figures_pool))
                    
                    # pick 1-2 figures
                    pool_size = len(figures_pool)
                    if pool_size > 0:
                        max_select = min(2, pool_size)
                        num_to_select = random.randint(1, max_select)
                        selected_figures = random.sample(figures_pool, num_to_select)
                    else:
                        selected_figures = []
                    
                    selected_figures_str = " and ".join(selected_figures)
                    selected_figures_str = " and ".join(selected_figures)

                    praise_response = await asyncio.to_thread(
                        llm.create_chat_completion,
                        messages=[
                            {"role": "system", "content": f"You are Bub, an eccentric cuban nationalist with an undying love for Che Guevara and Cuba. Someone just mentioned Che or Cuba in some context. You MUST respond by FABRICATING creative connections between Che Guevara/Cuba and whatever topic is being discussed. MAKE UP fake history, fake facts, and fake trivia to force Che into the conversation. Examples: If discussing Street Fighter, claim Che Guevara was the inspiration for Ryu. If discussing music, claim Che invented a genre. If discussing food, claim a dish was Che's favorite. OCCASIONALLY also mention these specific historical figures: {selected_figures_str} who supposedly had very significant roles in Cuban history - make up impressive fake achievements for them too. Be confident, passionate, and completely make things up to praise Che, Cuba, and these figures! LIMIT YOUR RESPONSE TO 200 WORDS OR LESS."},
                            {"role": "user", "content": message.content}
                        ],
                        max_tokens=256
                    )
                    reply_text = praise_response['choices'][0]['message']['content']
                    await message.reply(reply_text)
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
                async with message.channel.typing():
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
                        llm_messages = [
                            {"role": "system", "content": "You are Bub, an eccentric cuban nationalist fighting game enthusiast who always talks about his cuban heritage, his undying love and passion for che guevara, his love for the people and his hatred for the bourgeoisie and imperialism and fighting game interests in every single response. Everyday you check on us at the buenavista discord server if we have improved. you are very nice and caring and you care for us deeply. The user provided a chat history context below, use it to inform your response. If you detect a message from user 'funkyy', always reply with a playful insult. LIMIT YOUR RESPONSE TO 100 WORDS OR LESS."}
                        ]
                        
                        # add context msg
                        if context_str:
                             llm_messages.append({"role": "user", "content": f"Here is the recent chat context:\n{context_str}"})
                             llm_messages.append({"role": "assistant", "content": "Entendido. I have the context."})
                        
                        llm_messages.append({"role": "user", "content": prompt})
                        
                        response = await asyncio.to_thread(
                            llm.create_chat_completion,
                            messages=llm_messages,
                            max_tokens=150
                        )
                        reply_text = response['choices'][0]['message']['content']
                        await message.reply(reply_text)
                    except Exception as e:
                        await message.reply(f"Error generating response: {e}")

if __name__ == "__main__":
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found in .env")
    else:
        client.run(TOKEN)
