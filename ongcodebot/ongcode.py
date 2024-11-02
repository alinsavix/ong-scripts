#!/usr/bin/env python3
import argparse
import datetime
import os
import pprint
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import discord
import sqlalchemy
import toml
from sqlalchemy import (Column, DateTime, Float, Integer, String, Time,
                        create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker
from tdvutil import ppretty
from tdvutil.argparse import CheckFile

# aiosqlite

# Base = declarative_base()
# class OngcodeMessage(Base):
#     __tablename__ = "ongcode_messages"

# things we need to keep:
# oncode message id
# ongcode date
# ongcode edit date
# title message id
# title author
# title
# title date
# title edit date
#     id = Column(Integer, primary_key=True)
#     created_at = Column(DateTime)
#     content = Column(String)


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

    parser.add_argument(
        "--ongcode-server",
        type=str,
        default="WWP",
        help="Server to use for finding ongcode"
    )

    parser.add_argument(
        "--ongcode-channel",
        type=str,
        default="testing-private",
        help="channel to use for finding ongcode"
    )

    parsed_args = parser.parse_args()

    if parsed_args.credentials_file is None:
        parsed_args.credentials_file = Path(__file__).parent / "credentials.toml"

    return parsed_args


# make sure to also read https://guide.pycord.dev/getting-started/more-features
class OngcodeBot(discord.Bot):
    botchannel: discord.TextChannel
    def __init__(self, botargs: argparse.Namespace):
        self.botargs = botargs

        intents = discord.Intents.default()
        # intents.presences = True
        intents.messages = True
        intents.message_content = True
        intents.reactions = True
        # intents.typing = True
        intents.members = True

        super().__init__(intents=intents)   # , status=discord.Status.invisible)

    async def on_ready(self):
        # print(ppretty(self))
        print(f"{self.user} (id {self.user.id}) is online")

        print(f"finding channel #{self.botargs.ongcode_channel}")
        channel = discord.utils.get(self.get_all_channels(), guild__name=self.botargs.ongcode_server, name=self.botargs.ongcode_channel)

        if channel is None:
            log(f"ERROR: channel #{self.botargs.ongcode_channel} not found, can't reasonably continue")
            os._exit(1)

        print(f"found channel with id {channel.id}")
        self.botchannel = channel

        await self.message_catchup()

    async def on_message(self, message: discord.Message):
        if message.author.id == self.user.id:
            return

        print(f"message from {message.author}: {message.content}")
        print(message)

    async def message_catchup(self) -> None:
        datetime_now = datetime.datetime.now()

        print("catching up on messages...")
        async for h in self.botchannel.history(limit=2):  # , before=datetime_now):
            print("====================================")
            # print(ppretty(h))
            print(h.author.nick)
            print(h.channel.name)
            print(h.clean_content)
            print(h.id)
            print(h.thread)
            print(h.created_at)
            print(h.edited_at)

            # things we need to keep:
            # oncode message id
            # ongcode date
            # ongcode edit date
            # title message id
            # title author
            # title
            # title date
            # title edit date

        print("done")



def main():
    args = parse_args()
    creds = get_credentials(args.credentials_file, args.environment)
    # print(creds)

    bot = OngcodeBot(botargs=args)

    @bot.slash_command(name="hello", description="Say hello to the bot")
    async def hello(ctx: discord.ApplicationContext):
        await ctx.respond("Hey!")

    bot.run(creds["token"])

if __name__ == "__main__":
    main()
