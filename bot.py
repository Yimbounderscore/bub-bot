import discord
import os
import asyncio
import random
import datetime
import re
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

# Comprehensive list of affirmative and negative keywords/phrases
# Used for detecting user intent in replies.
POSITIVE_KEYWORDS = {
    "yes", "yeah", "yep", "yea", "yup", "sure", "absolutely", "definitely",
    "indeed", "affirmative", "certainly", "of course", "ok", "okay", "alright",
    "fine", "yessir", "yessirrie", "you bet", "aye", "ja", "si", "oui",
    "correct", "right", "roger", "copy that", "sure thing", "i have",
    "totally", "totes", "for sure", "gladly", "undoubtedly", "without a doubt",
    "hell yeah", "heck yeah", "positive", "agreed", "ye", "yur", "yuh"
}

NEGATIVE_KEYWORDS = {
    "no", "nah", "nope", "nay", "negative", "never", "not really", "not at all",
    "absolutely not", "no way", "doubt it", "unlikely", "pass", "no thanks",
    "i haven't", "fail", "failed", "worse", "terrible", "bad", "nop",
    "nah fam", "nuh uh", "hard pass", "lose", "lost"
}

# Large library of bot replies
ROTATING_RESPONSES = {
    "positive": [
        "awesomesauce", "Awesome!", "Great to hear!", "That's fantastic!", "Splendid!",
        "Excellent news!", "Keep it up!", "Proud of you!", "Way to go!", "Nice work!",
        "Brilliant!", "Outstanding!", "Marvelous!", "Wonderful!", "Superb!",
        "Glad to hear it!", "You're doing great!", "Amazing!", "Fantastic!", "Good vibes only!",
        "That's the spirit!", "You nailed it!", "Heck yeah!", "Boom!", "Score!",
        "W!", "Big W!", "Winning!", "Legendary!", "Epic!",
        "Top tier!", "Goated!", "Based!", "Slay!", "Queen behavior!",
        "King behavior!", "So true!", "Love that for you!", "Absolute cinema!", "Chef's kiss!",
        "On fire!", "Crushing it!", "Unstoppable!", "Beast mode!", "Let's gooo!",
        "So cool!", "Very nice!", "Radical!", "Tubular!", "Groovy!",
        "Spectacular!", "Phenomenal!", "Incredible!", "Unbelievable!", "Stupendous!",
        "Magnificent!", "Glorious!", "Triumphant!", "Victorious!", "Success!",
        "Mission accomplished!", "Target destroyed (in a good way)!", "Level up!", "Gains!", "Progress!",
        "Ascended!", "Transcended!", "Enlightened!", "Awakened!", "Blessed!",
        "Grateful!", "Thankful!", "Appreciative!", "Kind!", "Sweet!",
        "Top marks!", "First class!", "Premium!", "Elite!", "fucking nerd lmao",
        "Ascended!", "Transcended!", "Enlightened!", "Awakened!", "Blessed!",
        "Grateful!", "Thankful!", "Appreciative!", "Kind!", "Sweet!",
        "Lovely!", "Beautiful!", "Gorgeous!", "Stunning!", "Breathtaking!",
        # Fighter / FGC Updates
        "Perfect! (SF Announcer Voice)", "You win!", "K.O.!", "Shoryuken!", "Sonic Boom!",
        "Round 1... Fight!", "Hadouken!", "Combo breaker!", "Frame perfect!", "Plus on block!",
        "Okizeme on point!", "Super art finish!", "Critical Art!", "Legendary finish!", "Rank up!",
        "Godlike reads!", "Punish counter!", "Drive Rush cancel!", "Option select successful!",
        "Heaven or Hell, Let's Rock!", "Destroyed!", "Fatality!", "Brutality!", "Flawless Victory!",
        "Happy Birthday! (Hit two people)", "Evo Moment #37!", "Daigo Parry!", "Red Parry!", "Just Frame!",
        "Electric Wind God Fist!", "DORYA!", "Volcanic Viper!", "Buster Wolf!", "Are you okay? BUSTER WOLF!",
        "Touch of death!", "Infinite combo!", "Mix-up god!", "50/50 guesses correct!", "Reset city!",
        "Download complete!", "Read like a book!", "Techable!", "Clutch!", "Comeback mechanic activated!",
        "V-Trigger activated!", "Instinct Mode!", "Sparking!", "X-Factor Level 3!", "Roman Cancel!", "Burst bait!"
    ],
    "negative": [
        "oh hope you improve next time", "Oh no...", "That's unfortunate.", "Sad to hear that.", "Better luck next time.",
        "Hang in there.", "Don't give up.", "Tomorrow is a new day.", "It happens to the best of us.", "Keep your head up.",
        "Stay strong.", "You'll get 'em next time.", "Oof.", "Rip.", "F in chat.",
        "Unlucky.", "Rough day?", "Sending positive vibes.", "Hope it gets better.", "Deep breaths.",
        "One day at a time.", "This too shall pass.", "Storms make trees take deeper roots.", "Chins up!", "You got this.",
        "Believe in yourself.", "Don't be discouraged.", "Setbacks are setups for comebacks.", "Keep pushing.", "Stay resilient.",
        "Sadge.", "PepeHands.", "Not pog.", "Weirdchamp.", "Dang.",
        "Darn.", "Bummer.", "Look on the bright side.", "Could be worse!", "At least you're here.",
        "You're alive!", "You're breathing!", "Small steps.", "Progress is non-linear.", "Be kind to yourself.",
        "Treat yourself.", "Rest up.", "Recharge.", "Reset.", "Try again.",
        "Failure is part of success.", "Learning opportunity!", "Character building!", "Growth mindset!", "Pivot!", "build a solid effective flowchart with your character, dr jab more dosent matter the dr speed, build good fundies make sure u anti air constantly, crosscut consistently, make sure you know when to use your meter build routes vs cashout routes, make sure u know max damage routes, corner carry routes, learn just enough matchup knowlege to beat characters layer 1 stuff like fb dr or whatever nonsense u run into, replay review ever match that gives you trouble especially the ones that pissed you off, udont need ot drill anything but understanding why you lost is important and if you see a pattern in whats causing the loss you can fix it. dont switch characters either jsut pick one forget about tierlists or difficulty whatever is most fun to you and note that u dont need to beat 1800s you just need to beat enough 16-1700s to get out and especially if u are in the netherlands the queue is prob so dry u genuinley just need to download 3-4 1700+ players and the rest is 1600 fodder when ur trying to get 1800",
       
        # Fighter / FGC Updates
        "You lose.", "Continue? 10... 9...", "Go home and be a family man.", "Defeated.", "Rank down...",
        "Chip damage victory (for the enemy).", "Whiffed punish.", "Dropped the combo.", "Button masher.", "Scrub quote.",
        "Salt mining.", "Run it back.", "Perfect KO... against you.", "Stunned!", "Guard broken.",
        "Reversal failed.", "Wake-up DP failed.",
        "Double K.O.", "Time Over.", "Draw Game.", "You must defeat Sheng Long to stand a chance.",
        "Weak.", "Pathetic.", "Get over here!... and lose.", "Finish Him!", "Wasted.", "You Died.",
        "Command grab whiffed.", "Hard read failed.", "Mashed DP on wake-up.", "Did you learn how to block?",
        "Spammed fireballs and lost.", "Rage quit.", "Disconnected.", "Lag switch?", "Input delay.",
        "Misinput.", "No tech.", "Reset bracket.", "Eliminated.", "0-2 drop.", "Sent to losers bracket.",
        "Gatekeeper stopped you.", "Skill check failed.", "Knowledge check failed.", "Frame trapped.", "Counter hit!"
    ]
}

def is_affirmative(text):
    """Checks if the text contains any affirmative keywords."""
    cleaned = text.lower().strip()
    # Check for direct match or starts with (to catch "yes I did")
    if cleaned in POSITIVE_KEYWORDS:
        return True
    return any(cleaned.startswith(word + " ") or cleaned.startswith(word + ".") or cleaned.startswith(word + "!") for word in POSITIVE_KEYWORDS)

def is_negative(text):
    """Checks if the text contains any negative keywords."""
    cleaned = text.lower().strip()
    if cleaned in NEGATIVE_KEYWORDS:
        return True
    return any(cleaned.startswith(word + " ") or cleaned.startswith(word + ".") or cleaned.startswith(word + "!") for word in NEGATIVE_KEYWORDS)

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

    # Send messages immediately on startup
    await send_daily_messages(channel)

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

    # Easter egg: Jailbreak attempt response
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

    # Easter egg: Tarkus response
    if "tarkus" in message.content.lower():
        await message.reply("My brother is African American. Our love language is slurs and assaulting each other.")

    # Logic flags
    check_sentiment = False
    check_media = False
    
    # 1. Check Mentions
    if client.user.mentioned_in(message):
        check_sentiment = True
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
                # If replying to the specific question
                if replied_msg.content == "Has anyone improved?":
                    check_sentiment = True

        except discord.NotFound:
            pass
        except discord.Forbidden:
            pass
        except Exception as e:
            print(f"Reply logic error: {e}")

    # Execution phases
    # Phase A: Sentiment Check (Positive/Negative Text)
    if check_sentiment:
        # Strip mentions to handle "@BubBot yes" correctly
        # Regex removes <@123...> and <@!123...> patterns
        content_no_mentions = re.sub(r'<@!?[0-9]+>', '', message.content).strip()
        
        if is_affirmative(content_no_mentions):
            reply_text = random.choice(ROTATING_RESPONSES["positive"])
            await message.reply(f"{reply_text} <:sponge:1416270403923480696>")
        elif is_negative(content_no_mentions):
            reply_text = random.choice(ROTATING_RESPONSES["negative"])
            await message.reply(f"{reply_text} <:sponge:1416270403923480696>")

    # Phase B: Media Check (GIFs/Images)
    if check_media:
        # Check for GIF in embeds
        has_gif = any("tenor.com" in str(e.url or "") or "giphy.com" in str(e.url or "") or (e.type == "gifv") for e in message.embeds)
        # Check message content for GIF links
        if not has_gif:
            has_gif = "tenor.com" in message.content.lower() or "giphy.com" in message.content.lower() or ".gif" in message.content.lower()
        
        # Check for image attachments
        has_image = any(att.content_type and att.content_type.startswith("image/") for att in message.attachments)
        
        if has_gif or has_image:
            await message.reply("nice i like kpop! <:sponge:1416270403923480696>")

if __name__ == "__main__":
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found in .env")
    else:
        client.run(TOKEN)
