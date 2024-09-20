#!/usr/bin/env python3
import argparse
import pprint
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import discord
import toml
from tdvutil import ppretty
from tdvutil.argparse import CheckFile

# aiosqlite



def log(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.stderr.flush()

def now() -> int:
    return int(time.time())

def get_credentials(cfgfile: Path, environment: str) -> str:
    log(f"loading config from {cfgfile}")
    config = toml.load(cfgfile)

    try:
        return config["ongcode_bot"][environment]
    except KeyError:
        log(f"ERROR: no configuration for ongcode_bot.{environment} in credentials file")
        sys.exit(1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Share to discord some of the changes ojbpm has detected")

    parser.add_argument(
        "--credentials-file", "-c",
        type=Path,
        default=None,
        action=CheckFile(must_exist=True),
        help="file with discord credentials"
    )

    parser.add_argument(
        "--environment", "--env",
        type=str,
        default="test",
        help="environment to use"
    )

    parsed_args = parser.parse_args()

    if parsed_args.credentials_file is None:
        parsed_args.credentials_file = Path(__file__).parent / "credentials.toml"

    return parsed_args


# make sure to also read https://guide.pycord.dev/getting-started/more-features
class OngcodeBot(discord.Bot):
    def __init__(self):
        intents = discord.Intents.default()

        # intents.presences = True
        intents.messages = True
        intents.message_content = True
        intents.reactions = True
        # intents.typing = True
        intents.members = True

        super().__init__(intents=intents)   # , status=discord.Status.invisible)

    async def on_ready(self):
        print(f"{self.user} (id {self.user.id}) is online")

    async def on_message(self, message: discord.Message):
        if message.author.id == self.user.id:
            return

        print(f"message from {message.author}: {message.content}")
        print(message)


def main():
    args = parse_args()
    creds = get_credentials(args.credentials_file, args.environment)
    # print(creds)

    bot = OngcodeBot()

    @bot.slash_command(name="hello", description="Say hello to the bot")
    async def hello(ctx: discord.ApplicationContext):
        await ctx.respond("Hey!")

    bot.run(creds["token"])

if __name__ == "__main__":
    main()
