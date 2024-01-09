import datetime
import functools
import io
import json
import os
import pprint
import re
import time
import traceback
from collections import OrderedDict
import datetime
from typing import Any, Dict, Union

import discord
import openai
import requests
import selenium
from colorama import Fore, Style
from discord import Intents, Member
from discord.ext import commands
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from spellchecker import SpellChecker
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation")

spell = SpellChecker()
[spell.word_frequency.add(word) for word in ["SETX", "$val", "<valtracker>", "bruh"]]

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

test_channel_id = 1176446310015041618
admin_channel_id = 1180059589413183498

jet_id = 765462282079043628
wee_shen_id = 940251593780113530
jared_id = 1021668580481306684
benjamin_id = 932171028359180298
bryan_id = 883252262082850827

# Type hint for the "message" key
MessageOrNone = Union[discord.Message, None]

# Type hint for each file entry
FileEntry = Dict[str, Union[Dict[str, str], MessageOrNone, str, bool]]

# Type hint for the entire dictionary
NecessaryFilesDict = Dict[str, FileEntry]

necessary_files: NecessaryFilesDict = {
    "hyperparams": {"data": {}, "message": None, "filename": "hyperparams.json", "found": False},
    "openai_api": {"data": {}, "message": None, "filename": "openai_api.txt", "found": False},
    "function_permissions": {"data": {}, "message": None, "filename": "function_permissions.json", "found": False},
    "member_details": {"data": {}, "message": None, "filename": "member_details.json", "found": False},
}

bot_data_channel: discord.TextChannel = None

ready = False

timeout_duration = datetime.timedelta(seconds=60)

with open("function_details.json", "r") as f:
    function_details = json.load(f)

with open("function_permissions.json") as f:
    function_permissions = json.load(f)
with open("hyperparams.json") as f:
    hyperparams = json.load(f)


def print_red(text):
    print(Fore.RED + text + Style.RESET_ALL)


def neat_print(dictionary, heading1, heading2):
    # Determine the length of the longest word in each column
    len1 = max(len(heading1), max((len(str(key)) for key in dictionary.keys()), default=0))
    len2 = max(len(heading2), max((len(str(value)) for value in dictionary.values()), default=0))

    # Print the table headings
    print(f"{heading1:<{len1}} | {heading2:<{len2}}")
    print(f"-{'-' * len1}|{'-' * len2}-")

    # Print the dictionary items
    for key, value in dictionary.items():
        print(f"{str(key):<{len1}} | {str(value):<{len2}}")
    print(f"-{'-' * len1}|{'-' * len2}-")


def correct_spelling(text):
    spell_ = spell
    words = text.split()

    word_dict = {word: spell_.correction(word) for word in words}
    word_dict = OrderedDict(word_dict)

    uncorrectable_words = [word for word, correction in word_dict.items() if correction is None]
    # print("Uncorrectable words:", uncorrectable_words)
    #
    # neat_print(word_dict, "Original words", "Corrected words")

    corrected_text = ' '.join([word_dict[word] if word_dict[word] is not None else word for word in words])
    return corrected_text


class GPT:
    openai_api_key = "this is not a valid api key"

    @staticmethod
    def _Google_search(Query, k):
        search = GoogleSearchAPIWrapper(k=k)

        tool = Tool(
            name="I'm Feeling Lucky",
            description="Search Google and return the first result.",
            func=search.run,
        )

        return tool.run(Query)

    def run_convo(self, model, message, max_tokens=2000, temperature=1, image=None):
        global member_details
        input_text = message.content[5:]
        output_text = ""

        if model == "gpt-4-vision-preview":
            messages = [{"role": "user", "content": {
                {
                    "type": "text",
                    "text": message.content[5:]
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    },
                }
            }}]
        else:
            messages = [{"role": "user", "content": {
                {
                    "type": "text",
                    "text": message.content[5:]
                }
            }}]

        tools = [

            {"type": "function",
             "function": {
                 "name": "Google_search",
                 "description": "Do a google search, getting the first k results",
                 "parameters": {
                     "type": "object",
                     "properties": {
                         "Query": {
                             "type": "string",
                             "description": "Search query",
                         },
                         "k": {
                             "type": "integer",
                             "description": "Number of results",
                         },
                     },
                     "required": ["Query", "k"],
                 },

             }}]
        available_functions = {"Google_search": self._Google_search}

        try:

            client = OpenAI(api_key=self.openai_api_key)

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=temperature,
                tools=tools,
                tool_choice="auto"
            )

            price = response['usage']['total_tokens']

        except openai.AuthenticationError as e:

            print(e.message)
            return "There was an error authenticating with OpenAI."

        response_message = response.choices[0].message

        # Step 2: check if GPT wanted to call a function
        if response_message.tool_calls:
            try:
                # Step 3: call the function
                # Note: the JSON response may not always be valid; be sure to handle errors
                # only one function in this example, but you can have multiple

                messages.append(response_message)
                try:
                    input_text += response_message.content
                except:
                    pass

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)

                    print_red(f"AI called function {function_name}")
                    output_text += function_name.join(function_args)

                    # Call the function that the AI wants to call
                    function_response = function_to_call(**function_args)

                    # add func response into input text
                    input_text += function_response.join(function_name).join(tool_call.id)

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })

                second_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                    temperature=temperature,

                )  # get a new response from GPT where it can see the function response

                output_text += second_response.choices[0].message.content

                price += response['usage']['total_tokens']

                try:
                    member_details[str(message.author.id)]["spent on GPT"] += price
                except KeyError:
                    # Add the member if the ID does not exist
                    member_details[str(message.author.id)] = {"name:": message.author.global_name,
                                                              "spent on GPT": price}

                return f'{second_response.choices[0].message.content}\nPrice for this generation: {round(price, 2)}¢\nYour total usage: ${round(member_details[str(message.author.id)]["spent on GPT"], 2) / 100}'

            except Exception as e:

                print(f"Error:\n {traceback.format_exc()}")

                return "An error occurred. This is most likely caused by the incorrect output from the bot."

        else:
            output_text = response_message.content

            print(type(price))

            with open("member_details.json", "r") as f:
                member_details = json.load(f)

            member_details[str(message.author.id)]["spent on GPT"] = int(
                member_details[str(message.author.id)]["spent on GPT"]) + price

            with open("member_details.json", "w") as f:
                json.dump(member_details, f)

            return f'{response_message.content}\nPrice for this generation: {round(price, 2)}¢\nYour total usage: ${round(member_details[str(message.author.id)]["spent on GPT"], 2) / 100}'


class GetTrackerInfo:
    options = Options()
    # options.add_argument('--headless')
    options.add_argument("--start-fullscreen")
    # options.add_argument("--start-minimized")
    driver = None

    # total no of params
    Val_vars = 3

    def _upload_file_imgur(self, b64_img):

        # Your Imgur client ID
        client_id = os.environ.get("IMGUR_CLIENT_ID")

        # Headers for the API request
        headers = {'Authorization': 'Client-ID ' + client_id}

        # Data for the API request
        data = {'image': b64_img}

        # Make the API request
        response = requests.post('https://api.imgur.com/3/image', headers=headers, data=data)

        # Get the JSON response
        response_json = response.json()

        # Extract the link
        link = response_json['data']['link']

        pprint.pprint(response_json)

        return link

    def _wait_and_take_screenshot(self, path, by=By.CSS_SELECTOR, site=None):
        if site is None:
            pass
        else:
            self.driver.get(site)
            time.sleep(2)

        WebDriverWait(self.driver, 5).until(
            EC.presence_of_element_located((by, path)))

        element = self.driver.find_element(By.CSS_SELECTOR,
                                           path)

        b64_img = element.screenshot_as_base64
        return self._upload_file_imgur(b64_img)

    def _close_driver_and_return(self, response):
        self.driver.quit()
        return response

    def get_tracker_info(self, message, keywords, existing_members):
        self.driver = webdriver.Chrome(options=self.options)

        response = ""

        corrected_spelling = [correct_spelling(keyword).lower() for keyword in keywords]

        if not message.mentions:
            id_name_dict = {message.author.id: message.author.global_name}
        else:
            id_name_dict = {user.id: user.name for user in message.mentions}

        for id_, name in id_name_dict.items():
            try:
                if "playtime" in corrected_spelling:
                    return self._close_driver_and_return(self._wait_and_take_screenshot(
                        "#app > div.trn-wrapper > div.trn-container > div > main > div.content.no-card-margin > div.site-container.trn-grid.trn-grid--vertical.trn-grid--small > div.trn-grid__sidebar-left > div:nth-child(1) > div.playtime-summary.trn-card",
                        site=f"https://tracker.gg/valorant/profile/riot/{existing_members[str(id_)]['VALTRACKER']}/performance"))

                if "comp" in corrected_spelling or "competitive" in corrected_spelling or "ranked" in corrected_spelling:
                    self.driver.get(
                        f"https://tracker.gg/valorant/profile/riot/{existing_members[str((id_))]['VALTRACKER']}/overview")

                    WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR,
                                                        "#app > div.trn-wrapper > div.trn-container > div > main > div.content.no-card-margin > div.site-container.trn-grid.trn-grid--vertical.trn-grid--small > div.trn-grid.container > div.area-main > div.area-main-stats > div.performance-score.mt-4 > div.performance-score__container > div.performance-score__stats > div:nth-child(4)"))
                    )
                    time.sleep(2)
                elif "unranked" in corrected_spelling or "unrated" in corrected_spelling or "casual" in corrected_spelling:
                    self.driver.get(
                        f"https://tracker.gg/valorant/profile/riot/{existing_members[str(id_)]['VALTRACKER']}/overview?playlist=unrated")
                    WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR,
                                                        "#app > div.trn-wrapper > div.trn-container > div > main > div.content.no-card-margin > div.site-container.trn-grid.trn-grid--vertical.trn-grid--small > div.trn-grid.container > div.area-main > div.area-main-stats > div > div.main > div:nth-child(12)")))

                    time.sleep(2)

                else:
                    return self._close_driver_and_return("State a type of metric you are looking for (Comp or unrated)")

            except KeyError:
                response += f"{name} has no valorant tracker profile registered. Use `SETX <VALTRACKER> <IN GAME NAME> <RIOT ID>` to register"
                pprint.pprint(existing_members[str(id_)])

            except selenium.common.exceptions.TimeoutException:
                return self._close_driver_and_return(
                    "You must register your accounting with tracker.gg before using this bot")

            if "accuracy" in corrected_spelling:
                response += self._wait_and_take_screenshot(
                    "#app > div.trn-wrapper > div.trn-container > div > main > div.content.no-card-margin > div.site-container.trn-grid.trn-grid--vertical.trn-grid--small > div.trn-grid.container > div.area-sidebar > div.accuracy.trn-card.trn-card--bordered.area-accuracy")

            if "top weapons" in corrected_spelling:
                response += self._wait_and_take_screenshot(
                    "#app > div.trn-wrapper > div.trn-container > div > main > div.content.no-card-margin > div.site-container.trn-grid.trn-grid--vertical.trn-grid--small > div.trn-grid.container > div.area-sidebar > div.top-weapons.trn-card.trn-card--bordered.area-top-weapons")

            if "hs%" in corrected_spelling or "headshot%" in corrected_spelling or "overview" in corrected_spelling:
                response += self._wait_and_take_screenshot(
                    "#app > div.trn-wrapper > div.trn-container > div > main > div.content.no-card-margin > div.site-container.trn-grid.trn-grid--vertical.trn-grid--small > div.trn-grid.container > div.area-main > div.area-main-stats > div.card.bordered.header-bordered.responsive.segment-stats")

            if "top agents" in corrected_spelling:
                response += self._wait_and_take_screenshot(
                    "#app > div.trn-wrapper > div.trn-container > div > main > div.content.no-card-margin > div.site-container.trn-grid.trn-grid--vertical.trn-grid--small > div.trn-grid.container > div.area-main > div.top-agents.area-top-agents")

            if not response:
                response = self._wait_and_take_screenshot(
                    "#app > div.trn-wrapper > div.trn-container > div > main > div.content.no-card-margin > div.site-container.trn-grid.trn-grid--vertical.trn-grid--small > div.trn-grid.container > div.area-main > div.area-main-stats")

        return self._close_driver_and_return(response)


class MyClient(commands.Bot, GPT, GetTrackerInfo, discord.Client):
    def __init__(self, *, intents: Intents, command_prefix="!",
                 **options: Any):
        super().__init__(intents=intents, **options, command_prefix=command_prefix)

        # Define the classification pipeline
        self.classifier = pipeline("text-classification", model="KoalaAI/Text-Moderation")


client = MyClient(intents=intents)


async def has_permission(func):
    try:
        details = function_permissions[func.__name__]
    except KeyError:
        print_red(
            f"There is an error with the file function_permissions. {func.__name__} permissions are not found.")
        return False

    @functools.wraps(func)
    def predicate(ctx):
        print("in wrapper")
        # Get the position of the specified role
        permissions = None
        try:
            role_position = discord.utils.get(ctx.guild.roles, id=details["role_id"]).position
        except:
            role_position = ctx.guild.default_role
        try:
            permissions = ctx.author.guild_permissions.__getattribute__(details["permissions"])
        except:
            pass
        if permissions or any(role.position >= role_position for role in ctx.author.roles) or details[
            "role_id"] == bryan_id:
            return True

    return await discord.ext.commands.check(predicate)


def preserve_metadata(func):
    return func

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


def compare_and_remove_keys(dict1, dict2):
    keys_to_remove = [key for key in dict1.keys() if key not in dict2]

    for key in keys_to_remove:
        dict1.pop(key, None)

    return dict1


@client.event
async def on_ready():
    print('Logged on as ', client.user)


@client.event
async def on_command_error(ctx, error):
    if isinstance(error, discord.ext.commands.errors.CommandNotFound):
        await ctx.send('The command you tried to use does not exist.')
        return
    func_name = ctx.command.name

    if isinstance(error, discord.ext.commands.MissingRequiredArgument) or isinstance(error,
                                                                                     discord.ext.commands.BadArgument):
        await ctx.send(function_details[func_name]["Definition"])
    elif isinstance(error, discord.ext.commands.CheckFailure):  # this is the final elif
        min_role = discord.utils.get(ctx.guild.roles, id=hyperparams[func_name]["min_role_id"])
        await ctx.send(f'The command "{func_name}" can only be used by {min_role}.')


    else:
        # Handle other types of errors here

        await ctx.send(error)


def second_decorator(coro):
    print("function wrapped")

    @functools.wraps(coro)  # Important to preserve name because `command` uses it
    async def wrapper(*args, **kwargs):
        print('wrapped function called')
        return await coro(*args, **kwargs)

    return wrapper


@has_permission
@client.command()
async def commands(ctx):
    table = "Function | Definition\n---------|---------\n"
    for func, details in function_details.items():
        definition = details["Definition"]

        # Format the func to the correct length
        func = func + (8 - len(func)) * " "

        # Format the definition
        definition = "".join([f"{sentence}.\n         |" for sentence in definition.split(".")])

        # Add extra new line
        definition = definition + "\n"

        table += f"{func} | {definition}"
    await ctx.message.channel.send(f"```{table}```")

    return


@has_permission
@client.command()
async def val(ctx, *keywords):
    temp_message = await ctx.message.channel.send("Your request has been received. Processing...")

    await temp_message.edit(content=client.get_tracker_info(ctx.message, keywords, member_details))
    return


@has_permission
@client.command()
## needs to be changed to put data in discord chat
async def SETX(ctx, app_name, *params):
    if app_name.upper() == "VALTRACKER":
        if len(params) != hyperparams["Val_vars"]:
            await ctx.send("Use `SETX <VALTRACKER> <IN GAME NAME> <RIOT ID>` without the brackets to register")

        member_details[ctx.message.author.id]["VALTRACKER"] = f"{params[1]}%23{params[1]}"

        await ctx.send("Your val tracker has been saved")
        await save_json_files(member_details_message_id, "member_details.json", member_details)

    else:
        await ctx.send(f"{app_name.upper()} is not available")
    return


@has_permission
@client.command()
async def gpt3(ctx):
    temp_message = await ctx.message.channel.send("Your request has been received. Processing...")
    await ctx.message.channel.send(f"According to GPT: \n{client.run_convo('gpt-3.5-turbo', ctx.message)}")
    await temp_message.delete()


@has_permission
@client.command()
async def gpt4(ctx):
    temp_message = await ctx.message.channel.send("Your request has been received. Processing...")
    await ctx.message.channel.send(f"According to GPT: \n{client.run_convo('gpt-4-preview', ctx.message)}")
    await temp_message.delete()


@has_permission
@client.command()
async def gpt4_vision(ctx):
    if ctx.message.author != bryan_id:
        return
    temp_message = await ctx.message.channel.send("Your request has been received. Processing...")
    await ctx.message.channel.send(
        f"According to GPT: \n{client.run_convo('gpt-4-preview', ctx.message, image=ctx.message.attachments[0].image)}")
    await temp_message.delete()


@has_permission
@client.command()
async def OFF(ctx, feature: str):
    if feature == "bot_responses_fun":
        client.bot_responses_fun = False
        hyperparams["bot responses fun"] = False
        await ctx.send("bot responses fun turned OFF")

    message = necessary_files["hyperparams"]["message"]
    attachments = []
    for attachment in message.attachments:
        if attachment.filename != "hyperparams.json":
            file = await attachment.to_file()
            attachments.append(file)

    json_file = json.dumps(hyperparams, indent=4)
    attachments.append(discord.File(json_file, filename="hyperparams.json"))

    await message.delete()
    await bot_data_channel.send(files=attachments)


@has_permission
@client.command()
async def ON(ctx, feature: str):
    if ctx.channel.id == admin_channel_id:
        if feature == "bot_responses_fun":
            client.bot_responses_fun = True
            hyperparams["bot responses fun"] = True
            await ctx.send("bot responses fun turned ON")

    message = necessary_files["hyperparams"]["message"]
    attachments = []
    for attachment in message.attachments:
        if attachment.filename != "hyperparams.json":
            file = await attachment.to_file()
            attachments.append(file)

    json_file = json.dumps(hyperparams, indent=4)
    attachments.append(discord.File(json_file, filename="hyperparams.json"))

    await message.delete()
    await bot_data_channel.send(files=attachments)


@has_permission
@client.command(name='timeout', description='timeouts a user for a specific time')
async def timeout(ctx, member: Member, seconds: int = 0, minutes: int = 0, hours: int = 0, days: int = 0,
                  reason: str = None):
    duration = datetime.timedelta(seconds=seconds, minutes=minutes, hours=hours, days=days)
    await member.timeout(duration, reason=reason)

    await ctx.send(f'{member.mention} was timeouted for {duration}', ephemeral=True)


@has_permission
@client.command()
async def jokes(ctx):
    if ctx.message.content.startswith('hehe'):
        await ctx.message.channel.send("chelsea best Tottenham suck")
    elif ctx.message.content.startswith("n-word-pass"):
        await ctx.message.delete()
        await ctx.message.channel.send("Wee shen = nigger")

    ids_replyed = [user.id for user in ctx.message.mentions]

    if wee_shen_id in ids_replyed:
        # await asyncio.sleep(2)
        await ctx.message.channel.send("Nigger please respond")
        await ctx.message.channel.send("https://media.tenor.com/K9aR8Y9-H-YAAAAd/indian.gif")

    if jared_id in ids_replyed:
        # await asyncio.sleep(2)
        await ctx.message.channel.send("Uncle please respond")

    if benjamin_id in ids_replyed:
        # await asyncio.sleep(2)
        await ctx.message.channel.send("O lvl pls respond")

    if bryan_id in ids_replyed:
        await ctx.message.channel.send("https://i.imgur.com/Z0w2euC.jpeg")

    if jet_id in ids_replyed:
        await ctx.message.channel.send("https://media.tenor.com/Hn3sDZt3VyMAAAAC/ooops-family-guy.gif")


@has_permission
@client.command()
async def set_value(ctx, *args):
    path = [args[0], "data"] + list(args[1:-1])
    value = args[-1]
    current = necessary_files

    for key in path[:-1]:
        if key in current:
            current = current[key]
        else:
            await ctx.message.channel.send (f"Key path not found. Search stopped at {key}")
            return

    else:
        last_key = path[-1]
        if last_key in current:
            # Modify the value at the last key
            current[last_key] = value
            print(f"Value at key path {path} modified.")
        else:
            await ctx.message.channel.send(f"\"{last_key}\" in key path not found.")
            return

    try:

        await save_json_files(necessary_files[path[0]]["message"], necessary_files[path[0]]["filename"],
                              necessary_files[path[0]]['data'])
    except KeyError as e:
        raise e

    print("sending return message")
    await ctx.message.channel.send(f"Key at {path} has been set to {value}")


@has_permission
@client.command()
async def add_role(ctx, member: discord.Member, role: discord.Role):
    await member.add_roles(role)
    await ctx.send(f"Added {role.name} to {member.display_name}")


@has_permission
@client.command()
async def remove_role(ctx, member: discord.Member, role: discord.Role):
    await member.remove_roles(role)
    await ctx.send(f"Removed {role.name} from {member.display_name}")


async def save_json_files(old_message: discord.Message, filename, file_):
    attachments = []

    for attachment in old_message.attachments:

        if attachment.filename != filename:
            file = await attachment.to_file()
            attachments.append(file)

    json_file = io.StringIO(json.dumps(file_, indent=4))
    attachments.append(discord.File(json_file, filename=filename))

    message_ = await bot_data_channel.send(files=attachments)

    for key in necessary_files.keys():
        for attachment in old_message.attachments:
            if attachment.filename == necessary_files[key]["filename"]:
                necessary_files[key]["message"] = message_

    await old_message.delete()

    return message_


@client.event
async def on_member_update(before, after):
    # Check if the username has changed
    if before.nick != after.nick:

        print(f"{before.nick}'s nickname changed to {after.nick}")
        outputs = client.classifier(after.nick)
        print(outputs)
        if outputs[0]["label"] != "OK" and outputs[0]["score"] > 1 - hyperparams["strictness"] / 100:
            member = before
            try:
                await member.timeout(datetime.timedelta(seconds=hyperparams["timeout_duration"]))
                await before.edit(nick=before.nick)
            except discord.Forbidden:
                pass
            # Send a DM to the member
            finally:

                await member.send(
                    f"Your name {after.nick} was flagged by our moderation system. Please remember to "
                    "keep the server safe for everyone.")
                if necessary_files["hyperparams"]['data']["logging_channel_id"]:
                    await before.guild.get_channel(necessary_files["hyperparams"]['data']["logging_channel_id"]).send(
                        f'"{after.nick}" was flagged. User was {before.nick}')

        else:
            print(f"{outputs[0]['label']} score {round(outputs[0]['score'], 2)}")


@client.event
async def on_message(message):
    global ready, bot_data_channel, hyperparams, function_permissions, necessary_files, openai_api_message_id, function_permissions_message_id, member_details, member_details_message_id

    # This line is necessary to process commands.
    await client.process_commands(message)

    ctx = await client.get_context(message)

    # if int(ctx.guild.id) == 1131946536755007689:
    #     return

    if not ready:
        ready = True
        channel = discord.utils.get(ctx.guild.channels, name="bot-data")
        bot_data_channel = channel

        if channel is not None:
            bot_data = channel.history(limit=200)

            bot_data = [x async for x in bot_data]
        else:
            bot_data = None

        async def check_file_completeness(file_name, attachment, message, file_extension="json"):
            """
            Updates the file in discord if a key is missing. File extension should not be used
            :param file_name:
            :param attachment:
            :param message:
            :param file_extension: does not work
            :return:
            """
            # Open the default file and load its content
            with open(f"{file_name}.{file_extension}", "r") as f:
                default_file = json.load(f)

            # Read the attachment content using await
            attachment_content = await attachment.read()

            try:
                # Parse the JSON content from the attachment
                dict_from_discord = json.loads(attachment_content)

                dict_from_discord = compare_and_remove_keys(dict_from_discord, default_file)

                # Check if the keys match
                if default_file.keys() == dict_from_discord.keys():
                    print("All necessary keys (in function definition dictionary) are present.")
                    necessary_files[file_name]["data"] = dict_from_discord
                else:
                    # Merge missing keys from the default file into the attachment content
                    for key in default_file.keys() - dict_from_discord.keys():
                        dict_from_discord[key] = default_file[key]

                    # Update the data in necessary_files
                    necessary_files[file_name]["data"] = dict_from_discord

                    # Assuming save_json_files is a function that saves the content to a file
                    await save_json_files(message, f"{file_name}.{file_extension}", dict_from_discord)

                    print("All keys in function_permissions file are now present.")
                    necessary_files[file_name]["found"] = True

                necessary_files[file_name]["message"] = message

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

        for message in bot_data:
            for attachment in message.attachments:
                if attachment.filename == "hyperparams.json" and not necessary_files["hyperparams"]["found"]:
                    await check_file_completeness("hyperparams", attachment, message)
                elif attachment.filename == "openai_api.txt" and not necessary_files["openai_api"]["found"]:
                    necessary_files["openai_api_message"] = attachment.to_dict()
                    necessary_files["openai_api_message"]["openai_api_message_message"] = message
                    necessary_files["openai_api"]["found"] = True

                elif attachment.filename == "function_permissions.json" and not \
                        necessary_files["function_permissions"]["found"]:
                    await check_file_completeness("function_permissions", attachment, message)


                elif attachment.filename == "member_details.json" and not necessary_files["member_details"][
                    "found"]:
                    necessary_files["openai_api_message"] = attachment.to_dict()
                    necessary_files["openai_api_message"]["openai_api_message_message"] = message
                    necessary_files["openai_api"]["found"] = True

        if channel is not None and bot_data:
            print("Bot data channel exists...")


        else:
            print("Bot data channel does not exist...")
            if channel:
                pass
            else:
                overwrites = {
                    ctx.guild.default_role: discord.PermissionOverwrite(read_messages=False),
                    ctx.guild.me: discord.PermissionOverwrite(read_messages=True),
                }

                channel = await ctx.guild.create_text_channel("bot-data", overwrites=overwrites)

                channel = await ctx.guild.create_text_channel("bot-data")
            bot_data_channel = channel
            member_details = await member_ids(ctx.guild)
            print(member_details)
            files = [discord.File(filename) for filename in
                     ["hyperparams.json", "openai_api.txt", "function_permissions.json", "member_details.json"]]
            message_ = await channel.send(files=files)
            await save_json_files(message_, "member_details.json", member_details)

    # don't respond to ourselves
    if message.author == client.user:
        return

    # Don't respond in
    if message.channel.id == test_channel_id:
        # await save_json_files(message, "hyperparams.json", hyperparams)
        return

    if message.channel.id != test_channel_id:

        if message.content.startswith("!"):
            return
        # Automod
        outputs = client.classifier(message.content)
        if outputs[0]["label"] != "OK" and outputs[0]["score"] > 1 - hyperparams["strictness"] / 100:
            print(outputs)
            member = ctx.guild.get_member(message.author.id)
            try:
                await member.timeout(datetime.timedelta(seconds=hyperparams["timeout_duration"]))

            except Exception:
                pass
            # Send a DM to the member
            await message.delete()
            await member.send(
                f'Your message "{message.content}" was flagged by our moderation system. Please remember to '
                'keep the server safe for everyone.')
            if necessary_files["hyperparams"]['data']["logging_channel_id"]:
                await ctx.guild.get_channel(necessary_files["hyperparams"]['data']["logging_channel_id"]).send(f'"{message.content}" was flagged. User was {member.nick}')
        else:
            print(f"{outputs[0]['label']} score {round(outputs[0]['score'], 2)}")

        message.content = correct_spelling(message.content)

        if "nigger" not in message.content:
            message.content.replace("nig", "nigger")

        # Remove the IDs from the message content
        message.content = re.sub(r'<@(.*?)>', '', message.content).strip()

        print("Message content:", message.content)

        # Send Images URL, stuff like that
        if message.attachments:
            print("test")
            return_message = f"{message.author.global_name}:\n{message.content}"
            return_attachments = (' '.join(
                [attachment.url if not attachment.content_type.startswith("audio") else "" for attachment in
                 message.attachments]))
            print("Return Message:", return_message, "\n", return_attachments)

            if not return_attachments:
                return

            # await message.channel.send(return_message)
            await message.channel.send(return_attachments)

            return


async def member_ids(guild):
    global necessary_files
    # Load existing members from the JSON file

    # Fetch all members
    async for member in guild.fetch_members():
        if str(member.id) in necessary_files["member_details"]['data']:
            # Update the name if the member ID exists
            necessary_files["member_details"]["data"][str(member.id)]["global_name"] = member.global_name
            necessary_files["member_details"]["data"][str(member.id)]["name"] = member.name

        else:
            # Add the member if the ID does not exist
            necessary_files["member_details"]["data"][str(member.id)] = {"global_name": member.global_name.__str__(),
                                                                         "name": member.name,
                                                                         "spent on GPT": 0}

    return necessary_files["member_details"]["data"]


client.run(os.environ.get("DISCORD_BOT_TOKEN"))
