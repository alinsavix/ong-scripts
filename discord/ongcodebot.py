#!/usr/bin/env python
import argparse
import asyncio
import datetime
import io
import os
import pprint
import re
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import toml
from more_itertools import ichunked
from peewee import (SQL, AutoField, BigIntegerField, CharField, DateTimeField,
                    FloatField, IntegerField, Model, SqliteDatabase, TextField)
from playhouse.sqlite_ext import (FTS5Model, RowIDField, SearchField,
                                  SqliteExtDatabase)
from tdvutil import ppretty
from tdvutil.argparse import CheckFile

import discord
from discord.ext import pages

MATCH_LIMIT = 10
# aiosqlite

class OngCode(Model):
    mainmsg_id = BigIntegerField(primary_key=True, null=False)
    mainmsg_date = DateTimeField(index=True, null=False)  # might need edit date too?
    mainmsg_text = TextField(null=False)

    titlemsg_id = BigIntegerField(index=True, null=True)  # should this be unique?
    titlemsg_author_id = BigIntegerField(null=True)
    titlemsg_author_name = CharField(null=True)
    titlemsg_date = DateTimeField(null=True)  # might need edit date too?
    titlemsg_text = TextField(null=True)

    class Meta:
        table_name = "ongcode"

class OngCodeIndex(FTS5Model):
    rowid = RowIDField()
    title = SearchField()

    class Meta:
        table_name = "ongcodeindex"
        options = {"tokenize": "unicode61"}

class OngCodeMeta(Model):
    key = CharField(primary_key=True)
    value = CharField()

    class Meta:
        table_name = "ongcodemeta"


_db: SqliteExtDatabase
_db_initialized = False
def initialize_db(dbfile: Path):
    global _db_initialized
    if _db_initialized:
        return

    log(f"INFO: Using database file {dbfile}")

    global _db
    _db = SqliteExtDatabase(None)
    _db.init(dbfile, pragmas={"journal_mode": "wal", "cache_size": -1 * 64 * 1024})
    _db.bind([OngCode, OngCodeIndex, OngCodeMeta])
    _db.connect()
    _db.create_tables([OngCode, OngCodeIndex, OngCodeMeta])

    _db_initialized = True


def get_db() -> SqliteExtDatabase:
    return _db


def set_ongcode_meta(key: str, value: str):
    meta = OngCodeMeta.replace(
        key=key,
        value=value
    )
    meta.execute()


def get_ongcode_meta(key: str) -> Optional[str]:
    m = OngCodeMeta.get_or_none(OngCodeMeta.key == key)
    if m:
        return m.value

    # else
    return None


def log(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.stderr.flush()


def now() -> int:
    return int(time.time())


id_start_re = re.compile(r"^\s*\^+\s*")
id_end_re = re.compile(r"\s*\^+\s*$")

bot_guild = None
bot_channel = None
mod_rolename = None

# make sure to also read https://guide.pycord.dev/getting-started/more-features
class OngcodeBot(discord.Bot):
    botchannel: discord.TextChannel
    last_nonid_msg_id: int
    last_nonid_msg_date: datetime.datetime
    caught_up: bool = False

    def __init__(self, botargs: argparse.Namespace):
        self.botargs = botargs

        lnm_id = get_ongcode_meta("last_nonid_msg_id")
        lnm_date = get_ongcode_meta("last_nonid_msg_date")
        assert lnm_id is not None and lnm_date is not None

        self.last_nonid_msg_id = int(lnm_id)
        self.last_nonid_msg_date = datetime.datetime.fromisoformat(lnm_date)

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
        log(f"{self.user} (id {self.user.id}) is online")

        log(f"finding channel #{self.botargs.ongcode_channel}")
        channel = discord.utils.get(self.get_all_channels(), guild__name=self.botargs.ongcode_guild, name=self.botargs.ongcode_channel)

        if channel is None:
            log(f"ERROR: channel #{self.botargs.ongcode_channel} not found, can't reasonably continue")
            os._exit(1)

        # print(ppretty(channel))

        log(f"found channel with id {channel.id}")
        self.botchannel = channel
        global bot_channel, bot_guild
        bot_channel = channel
        bot_guild = channel.guild

        await self.message_catchup()
        sys.stdout.flush()


    async def on_message(self, message: discord.Message):
        if message.channel.id != self.botchannel.id:
            return

        if message.author.id == self.user.id:
            return

        if not self.caught_up:
            return

        log(f"INFO: message from {message.author.nick}: {message.content}")
        self.process_message(message)


    async def message_catchup(self) -> None:
        log("catching up on messages...")
        log(f"Last recorded non-identifying message: {self.last_nonid_msg_date} (id {self.last_nonid_msg_id})")

        total_count = 0
        while True:
            count = 0

            async for h in self.botchannel.history(after=self.last_nonid_msg_date, limit=50, oldest_first=True):
                self.process_message(h)
                count += 1

            total_count += count
            if count == 0:
                log(f"INFO: caught up on (and indexed) {total_count} total messages")
                self.caught_up = True
                return

            # Otherwise, we need to get more messages to process
            log(f"INFO: caught up on {count} messages, looking for more...")
            await asyncio.sleep(5)


    def process_message(self, msg: discord.Message) -> None:
        # log(ppretty(msg))
        # Make sure we update the last processed timestamp appropriately
        self.last_nonid_msg_date = msg.created_at
        set_ongcode_meta("last_nonid_msg_date", str(self.last_nonid_msg_date))

        maybestr = ""
        if id_start_re.search(msg.clean_content) or id_end_re.search(msg.clean_content):
            # make sure there's something left after stripping everything out
            maybestr = re.sub(r"(\s|\^|\n|@\S+)+", "", msg.clean_content)

        if maybestr and "@JonathanOng" not in msg.clean_content:
            self.process_id_message(msg)
        else:
            self.process_ongcode_message(msg)


    def process_id_message(self, msg: discord.Message) -> None:
        if msg.type == discord.MessageType.reply:
            parentmsg = msg.reference.message_id
        else:
            if not self.last_nonid_msg_id:
                log("WARNING: Message looks like a song title, but no previous unknonw ongcode")
                log(f"WARNING: Message: {msg.clean_content}")
                return
            else:
                parentmsg = self.last_nonid_msg_id

        ongcode = OngCode.get_or_none(OngCode.mainmsg_id == parentmsg)
        if ongcode is None:
            log(f"WARNING: Previous ongcode message id {parentmsg} doesn't exist in database!")
            return

        ongcode.titlemsg_id = msg.id
        ongcode.titlemsg_author_id = msg.author.id
        ongcode.titlemsg_author_name = msg.author.nick or msg.author.name
        ongcode.titlemsg_date = msg.created_at
        ongcode.titlemsg_text = id_start_re.sub("", id_end_re.sub("", msg.clean_content))
        ongcode.save()

        idx = OngCodeIndex.get_or_none(OngCodeIndex.rowid == ongcode.mainmsg_id)
        if idx is not None:
            idx.title = ongcode.titlemsg_text
            idx.save()
        else:
            OngCodeIndex.create(
                rowid=ongcode.mainmsg_id,
                title=ongcode.titlemsg_text
            )

        log(f"INFO: saved ongcode identifier '{ongcode.titlemsg_text}' for ongcode message {ongcode.mainmsg_id}")

        # reset so that we don't process the same ongcode message twice
        # We'll keep the date, though, so that we know where to continue
        # from if we restart the bot
        if parentmsg == self.last_nonid_msg_id:
            self.last_nonid_msg_id = 0
            set_ongcode_meta("last_nonid_msg_id", str(self.last_nonid_msg_id))


    # FIXME: Might need further verification of some type
    def process_ongcode_message(self, msg: discord.Message) -> None:
        if len(msg.clean_content) < 50:
            log("WARNING: Message is probably too short to be ongcode")
            log(f"WARNING: Message: {msg.clean_content}")
            return

        OngCode.create(
            mainmsg_id=msg.id,
            mainmsg_date=msg.created_at,
            mainmsg_text=msg.clean_content
        )

        self.last_nonid_msg_id = int(msg.id)
        set_ongcode_meta("last_nonid_msg_id", str(self.last_nonid_msg_id))

        log(f"INFO: saved probable ongcode message with id {msg.id}")


def get_credentials(cfgfile: Path, environment: str) -> Dict[str, str]:
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
        "--ongcode-guild",
        type=str,
        default=None,
        help="Discord guild (server) to use for finding ongcode"
    )

    parser.add_argument(
        "--ongcode-channel",
        type=str,
        default=None,
        help="channel to use for finding ongcode"
    )

    parser.add_argument(
        "--moderator-role",
        type=str,
        default=None,
        help="role name for moderators"
    )

    parser.add_argument(
        "--dbfile",
        type=Path,
        default=None,
        help="database file to use"
    )

    parser.add_argument(
        "--debug-queries",
        default=False,
        action="store_true",
        help="print all queries to stderr",
    )

    parsed_args = parser.parse_args()

    if parsed_args.credentials_file is None:
        parsed_args.credentials_file = Path(__file__).parent / "credentials.toml"

    return parsed_args


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8", line_buffering=True)

    args = parse_args()
    creds = get_credentials(args.credentials_file, args.environment)
    args.creds = creds  # Store credentials in args

    if args.debug_queries:
        import logging
        logger = logging.getLogger('peewee')
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

    if args.ongcode_guild is None:
        args.ongcode_guild = creds.get("guild", None)

    if args.ongcode_guild is None:
        log("ERROR: No guild specified and no guild in configuration")
        sys.exit(1)

    if args.ongcode_channel is None:
        args.ongcode_channel = creds.get("channel", None)

    if args.ongcode_channel is None:
        log("ERROR: No channel specified and no channel in configuration")
        sys.exit(1)

    if args.moderator_role is None:
        args.moderator_role = creds.get("moderator_role", None)

    if args.dbfile is None:
        args.dbfile = Path(__file__).parent / f"ongcode_{args.environment}.db"

    log("INFO: In startup")
    log(f"INFO: Using guild '{args.ongcode_guild}'")
    log(f"INFO: Using channel '{args.ongcode_channel}'")

    initialize_db(args.dbfile)

    # See if we have a record for the last
    if not get_ongcode_meta("last_nonid_msg_date"):
        lnm_date = datetime.datetime(2016, 1, 1, 0, 0, 0)  # before ongcode
        set_ongcode_meta("last_nonid_msg_date", str(lnm_date))
        set_ongcode_meta("last_nonid_msg_id", str(0))

    bot = OngcodeBot(botargs=args)

    # @bot.slash_command(name="ping", description="Ping the bot.")
    # async def cmd_ping(ctx):
    #     await ctx.respond(f"Pong! Latency is {bot.latency * 1000:.1f}ms")

    # @discord.ext.commands.has_role(args.moderator_role)
    @bot.slash_command(name="ongcode", description="Search for ongcode")
    async def cmd_find_ongcode(
        ctx: discord.ApplicationContext,
        title: discord.Option(str, "Partial song title"),
    ):
        # await ctx.trigger_typing()
        # await asyncio.sleep(1)
        log(f"SEARCH: '{title}'")

        # Save ourselves some grief with special characters
        title = FTS5Model.clean_query(title)

        # this query was SO incredibly painful to figure out. Hint: You can't
        # use OngCodeIndex.search() here, even though the docs for peewee's
        # FTS5Model only list .search() as a class method. That's effectively
        # an entire query unto itself, though, so we have to use match() and
        # do the ranking ourselves
        q = (
            OngCode
            .select(OngCode, OngCodeIndex.rank().alias("score"))
            .join(OngCodeIndex, on=(OngCode.mainmsg_id == OngCodeIndex.rowid))
            .where(OngCodeIndex.match(title))
            .order_by(OngCodeIndex.rank())
        )

        log(f"SEARCH RESULT COUNT: {len(q)}")

        if discord.utils.get(ctx.author.roles, name=args.moderator_role):
            is_mod = True
        else:
            is_mod = False

        if len(q) == 0:
            embed = discord.Embed(
                title="Ongcode Search",
                # description="I'm the Ongcode bot. I'm here to help you find ongcode in the channel",
                color=discord.Color.red()
            )
            embed.add_field(name="", value="No matches found", inline=False)
            await ctx.respond(embed=embed, ephemeral=True)
            return

        pagelist = []

        for i, chunk in enumerate(ichunked(q, MATCH_LIMIT)):
            response_all = f"### Ongcode Search Results (page {i+1} of {(len(q) // MATCH_LIMIT) + 1})\n"
            for row in chunk:
                rowdate = datetime.datetime.fromisoformat(str(row.mainmsg_date)).date()

                msg_url = f"https://discord.com/channels/{bot_guild.id}/{bot_channel.id}/{row.mainmsg_id}"
                if is_mod:
                    response = f"{msg_url} - `{rowdate}` - {row.titlemsg_text} (score: {abs(row.score):.2f})\n"
                else:
                    response = f"`{rowdate}` - {row.titlemsg_text} (score: {abs(row.score):.2f})\n"

                response_all += response

            pagelist.append(
                pages.Page(content=response_all)
            )

        # Send the response
        paginator = pages.Paginator(pages=pagelist, disable_on_timeout=True, timeout=600)
        await paginator.respond(ctx.interaction, ephemeral=True)

        sys.stdout.flush()

    @cmd_find_ongcode.error
    async def cmd_find_ongcode_error(ctx: discord.ApplicationContext, error: discord.DiscordException):
        if isinstance(error, discord.ext.commands.MissingRole):
            await ctx.respond("Permission denied", ephemeral=True)

    @bot.slash_command(name="oc", description="Search for ongcode (alias for /ongcode)")
    async def cmd_find_ongcode_alias(
        ctx: discord.ApplicationContext,
        title: discord.Option(str, "Partial song title")
    ):
        await cmd_find_ongcode(ctx, title)

    @cmd_find_ongcode_alias.error
    async def cmd_find_ongcode_alias_error(ctx: discord.ApplicationContext, error: discord.DiscordException):
        if isinstance(error, discord.ext.commands.MissingRole):
            await ctx.respond("Permission denied", ephemeral=True)


    # Message command for sending ongcode to Jon
    @bot.message_command(name="Send Ongcode to Jon")
    async def cmd_ongcode_send(
        ctx: discord.ApplicationContext, message: discord.Message
    ):
        # Right channel?
        if message.channel.id != bot.botchannel.id:
            return

        # Not me?
        if message.author.id == bot.user.id:
            return

        if discord.utils.get(ctx.author.roles, name=bot.botargs.moderator_role):
            is_mod = True
        else:
            is_mod = False

        if not is_mod:
            await ctx.respond("Permission denied", ephemeral=True)
            return

        # Send initial response
        response = await ctx.respond("Processing your request...", ephemeral=True)
        message_obj = await response.original_message()

        # Check if this message is a title or ongcode
        ongcode = OngCode.get_or_none(
            (OngCode.mainmsg_id == message.id) | (OngCode.titlemsg_id == message.id)
        )

        if ongcode is None:
            await message_obj.edit(content="This message is not a recognized ongcode or title.")
            return

        # Get the title and body
        if message.id == ongcode.titlemsg_id:
            # This is a title message, so the body is in the main message
            title = ongcode.titlemsg_text
            body = ongcode.mainmsg_text
        else:
            # This is the main message, so we need to find the title
            title = ongcode.titlemsg_text or "Untitled"
            body = ongcode.mainmsg_text

        # Get the ongcodething endpoint from credentials
        ongcodething_endpoint = bot.botargs.creds.get("ongcodething_endpoint")
        if not codething_endpoint:
            await message_obj.edit(content="Error: ongcodething_endpoint not configured")
            return

        # Send to codething backend
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{ongcodething_endpoint}/songs/",
                    json={
                        "title": title,
                        "body": body,
                        "status": "PENDING"
                    }
                ) as response:
                    if response.status == 200:
                        await message_obj.edit(content=f"Successfully sent ongcode to Jon!\nTitle: {title}")
                    else:
                        await message_obj.edit(content=f"Failed to send ongcode: HTTP {response.status}")
            except Exception as e:
                await message_obj.edit(content=f"Error sending ongcode: {str(e)}")

        # Keep existing debug printing
        # print(ppretty(ctx))
        # print(ppretty(message))
        # print(f"ZOT: {ctx.author.id}")
        # print(f"ZOT: {message.author.id}")


    # A couple of testing things
    class MyModal(discord.ui.Modal):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            self.add_item(discord.ui.InputText(label="Short Input"))
            self.add_item(discord.ui.InputText(
                label="Long Input", style=discord.InputTextStyle.long))

        async def callback(self, interaction: discord.Interaction):
            embed = discord.Embed(title="Modal Results")
            embed.add_field(name="Short Input", value=self.children[0].value)
            embed.add_field(name="Long Input", value=self.children[1].value)
            await interaction.response.send_message(embeds=[embed])

    class MyView(discord.ui.View):
        @discord.ui.button(label="Send Modal")
        async def button_callback(self, button, interaction):
            await interaction.response.send_modal(MyModal(title="Modal via Button"))

    # @bot.slash_command()
    # async def send_modal(ctx):
    #     await ctx.respond(view=MyView())

    # creates a global message command. use guild_ids=[] to create guild-specific commands.
    # @bot.message_command(name="interaction_test")
    async def interaction_test(ctx, message: discord.Message):  # message commands return the message
        modal = MyModal(title="Modal via Message Command")
        await ctx.send_modal(modal)


    bot.run(creds["token"])


if __name__ == "__main__":
    main()
