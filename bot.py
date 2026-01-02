import discord
import os
import asyncio
import random
import datetime
import re
from dotenv import load_dotenv
from llama_cpp import Llama

# Load environment variables
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
try:
    CHANNEL_ID = int(os.getenv('CHANNEL_ID'))
except (TypeError, ValueError):
    print("Error: CHANNEL_ID not found or invalid in .env")
    CHANNEL_ID = None

# Intent setup - message_content required to read messages where bot isn't mentioned
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Check if model exists before loading to avoid errors
# Check if model exists before loading to avoid errors
MODEL_PATH = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
if not os.path.exists(MODEL_PATH):
    # Fallback to absolute path for linux server if working directory is wrong
    MODEL_PATH = "/opt/discordbot/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

llm = None
if os.path.exists(MODEL_PATH):
    print(f"Loading LLM model from {MODEL_PATH}...")
    try:
        # n_ctx=2048 limits context to save RAM. n_threads=4 for good CPU usage.
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)
        print("LLM loaded successfully.")
    except Exception as e:
        print(f"Failed to load LLM: {e}")
else:
    print(f"LLM model not found at {MODEL_PATH}. Chat feature disabled.")


async def send_daily_messages(channel):
    """Sends the standard batch of daily messages to the given channel."""
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
            # Introduce a throttle to value rate limits and ensure natural ordering.
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

    # Loop to schedule daily messages

    while not client.is_closed():
        now = datetime.datetime.now()
        
        # Schedule the next execution window.
        # Logic:
        # 1. Select a random timestamp within the 24-hour window of the current day.
        # 2. If the selected timestamp has already passed (Target < Now), schedule for the same relative timestamp on the following day.
        # 3. Calculate the delta between the target timestamp and the current time.
        
        # Normalize to the start of the current day (00:00:00).
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Generate a random offset in seconds (0 to 86399).
        random_seconds = random.randint(0, 86399)
        target_time = start_of_day + datetime.timedelta(seconds=random_seconds)

        if target_time < now:
            # The randomized slot for the current day has elapsed. Schedule for the next day.
            start_of_tomorrow = start_of_day + datetime.timedelta(days=1)
            random_seconds_tomorrow = random.randint(0, 86399)
            target_time = start_of_tomorrow + datetime.timedelta(seconds=random_seconds_tomorrow)
            print(f"Daily slot elapsed. Scheduling for next cycle at {target_time}")
        else:
            print(f"Scheduling for current cycle at {target_time}")

        # Calculate execution delay.
        wait_seconds = (target_time - datetime.datetime.now()).total_seconds()
        
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        # Execute message dispatch.
        await send_daily_messages(channel)
        
        # Post-execution cleanup:
        # Prevent immediate re-execution if the loop cycle completes rapidly.
        # Force a wait period until the start of the next calendar day (00:00:00) before regenerating the random schedule.
        
        # Calculate time remaining until the next midnight.
        next_day = (datetime.datetime.now() + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_until_tomorrow = (next_day - datetime.datetime.now()).total_seconds()
        print(f"Done for today. Waiting {seconds_until_tomorrow/3600:.2f} hours until midnight regeneration.")
        if seconds_until_tomorrow > 0:
            await asyncio.sleep(seconds_until_tomorrow)


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    # Start background task
    client.loop.create_task(background_task())

@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Check if the bot is mentioned and the phrase is present
    if client.user.mentioned_in(message) and "do the thing" in message.content.lower():
        await send_daily_messages(message.channel)

    # LLM Chat Trigger: "bub, [prompt]"
    if message.content.lower().startswith("bub, "):
        prompt = message.content[5:].strip() # Remove "bub, " prefix
        if llm:
            async with message.channel.typing():
                try:
                    # Run generation in a separate thread to avoid blocking the bot
                    response = await asyncio.to_thread(
                        llm.create_chat_completion,
                        messages=[
                            {"role": "system", "content": "You are Bub, a helpful and slightly chaotic Discord bot."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=256 # Limit output length
                    )
                    reply_text = response['choices'][0]['message']['content']
                    await message.reply(reply_text)
                except Exception as e:
                    await message.reply(f"Error generating response: {e}")
        else:
            await message.reply("I don't have a brain right now (Model file missing).")

    
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

    # Che Guevara / Cuba praise trigger (responds to any mention, not just @mentions)
    content_lower = message.content.lower()
    if "che guevara" in content_lower or "che " in content_lower or "cuba" in content_lower or " che" in content_lower:
        if llm:
            async with message.channel.typing():
                try:
                    praise_response = await asyncio.to_thread(
                        llm.create_chat_completion,
                        messages=[
                            {"role": "system", "content": "You are Bub, an eccentric cuban nationalist with an undying love for Che Guevara and Cuba. Someone just mentioned Che or Cuba. You MUST respond with passionate praise about the glory of Cuba, the heroism of Che Guevara, the revolution, and the fight against imperialism. Be enthusiastic and patriotic!"},
                            {"role": "user", "content": message.content}
                        ],
                        max_tokens=256
                    )
                    reply_text = praise_response['choices'][0]['message']['content']
                    await message.reply(reply_text)
                except Exception as e:
                    print(f"Che/Cuba praise error: {e}")

    # Logic flags
    check_media = False
    
    # 1. Check Mentions
    if client.user.mentioned_in(message):
        check_media = True

    # 2. Check Replies (Overrides/Adds to mention checks)
    if message.reference:
        try:
            if message.reference.cached_message:
                replied_msg = message.reference.cached_message
            else:
                replied_msg = await message.channel.fetch_message(message.reference.message_id)
            
            # If replying to the bot
            if replied_msg.author == client.user:
                check_media = True # Check media on any reply to bot

        except discord.NotFound:
            pass
        except discord.Forbidden:
            pass
        except Exception as e:
            print(f"Reply logic error: {e}")


    # Phase B: Media Check (GIFs/Images)
    media_found = False
    if check_media:
        # Check for GIF in embeds
        has_gif = any("tenor.com" in str(e.url or "") or "giphy.com" in str(e.url or "") or (e.type == "gifv") for e in message.embeds)
        # Check message content for GIF links
        if not has_gif:
            has_gif = "tenor.com" in message.content.lower() or "giphy.com" in message.content.lower() or ".gif" in message.content.lower()
        
        # Check for image attachments
        has_image = any(att.content_type and att.content_type.startswith("image/") for att in message.attachments)
        
        if has_gif or has_image:
            media_found = True
            await message.reply("nice i like kpop! <:sponge:1416270403923480696>")
            return # Stop processing execution phases

    # Phase B: LLM Response (If mentioned)
    if client.user.mentioned_in(message) and not media_found:
        content_no_mentions = re.sub(r'<@!?[0-9]+>', '', message.content).strip()
        prompt = content_no_mentions
        if prompt and llm:
                async with message.channel.typing():
                    try:
                        # Fetch previous message for context
                        context_msg = ""
                        async for prev_msg in message.channel.history(limit=2, before=message):
                            if prev_msg.author != client.user:  # Skip bot's own messages
                                context_msg = f"[Previous message from {prev_msg.author.display_name}: {prev_msg.content}]"
                                break
                        
                        # Build messages list with optional context
                        llm_messages = [
                            {"role": "system", "content": "You are Bub, an eccentric cuban nationalist fighting game enthusiast who always talks about his cuban heritage, his undying love and passion for che guevara, his love for the people and his hatred for the bourgeoisie and imperialism and fighting game interests in every single response. Everyday you check on us at the buenavista discord server if we have improved. you are very nice and caring and you care for us deeply. If context from a previous message is provided, use it to inform your response only if relevant. If you detect a message from user 'funkyy', always reply with a playful insult."}
                        ]
                        
                        # Add context as a separate message if found
                        if context_msg:
                            llm_messages.append({"role": "user", "content": context_msg})
                            llm_messages.append({"role": "assistant", "content": "Okay, I'll keep that context in mind."})
                        
                        llm_messages.append({"role": "user", "content": prompt})
                        
                        response = await asyncio.to_thread(
                            llm.create_chat_completion,
                            messages=llm_messages,
                            max_tokens=256
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
