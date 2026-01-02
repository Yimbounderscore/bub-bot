import discord
import os
import asyncio
import random
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
try:
    CHANNEL_ID = int(os.getenv('CHANNEL_ID'))
except (TypeError, ValueError):
    print("Error: CHANNEL_ID not found or invalid in .env")
    CHANNEL_ID = None

# Intent setup
intents = discord.Intents.default()
client = discord.Client(intents=intents)

async def background_task():
    await client.wait_until_ready()
    channel = client.get_channel(CHANNEL_ID)
    if not channel:
        print(f"Could not find channel with ID {CHANNEL_ID}")
        return

    print("Bot is ready and scheduling started.")

    # Send messages immediately on startup
    print("Sending messages immediately...")
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
            print(f"Error sending message: {e}")
    print("Initial messages sent!")

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

if __name__ == "__main__":
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found in .env")
    else:
        client.run(TOKEN)
