import pyperclip

def remove(data, text):
    return data.replace(text, "#" + text)\


massive_chunk = """  # Automod
        outputs = client.classifier(message.content)
        if outputs[0]["label"] != "OK" and outputs[0]["score"] > 1 - NecessaryFiles.hyperparams.data[
            "strictness"] / 100:
            print(outputs)
            member = ctx.guild.get_member(message.author.id)
            try:
                await member.timeout(datetime.timedelta(seconds=NecessaryFiles.hyperparams.data["timeout_duration"]))

            except Exception:
                pass
            # Send a DM to the member
            await message.delete()
            await member.send(
                f'Your message "{message.content}" was flagged by our moderation system. Please remember to '
                'keep the server safe for everyone.')
            if NecessaryFiles.hyperparams.data["logging_channel_id"]:
                await ctx.guild.get_channel(int(NecessaryFiles.hyperparams.data["logging_channel_id"])).send(
                    f'"{message.content}" was flagged. User was {member.nick}')

            print(f'"{message.content}" was flagged. User was {member.nick}')
        else:
            print(f"{outputs[0]['label']} score {round(outputs[0]['score'], 2)}")"""


with open("bot_refined.py", "r") as f:
    data = f.read()
    data = remove(data,"from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline")
    data = remove(data,"from spellchecker import SpellChecker")
    data = remove(data, 'model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation")')
    data = remove(data, 'tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")')
    data = remove(data, 'self.classifier = pipeline("text-classification", model="KoalaAI/Text-Moderation")')
    data = data.replace('outputs = client.classifier(after.nick)', "raise NotImplementedError")
    data = data.replace('client.run(os.environ.get("DISCORD_BOT_TOKEN"))',
                        'client.run("MTE3NjQ0ODIzOTM5OTA3NTkxMA.G2Y4R-.5vWMTXvjoVyXLr-GwCj0lgMJBcR2tE2I9LA_bg")')

    # for line in massive_chunk.split("\n"):
    #     data = remove(data, line)
    data = data.replace(massive_chunk, "")
# print(data)
with open("no_automod.py", "w") as f:
    f.write(data)

pyperclip.copy(data)