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
from rapidfuzz import fuzz, process
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


def load_title_cache() -> List[tuple[int, str]]:
    """Load all titles from the database into memory for fuzzy matching."""
    query = OngCode.select(OngCode.mainmsg_id, OngCode.titlemsg_text).where(
        OngCode.titlemsg_text.is_null(False)
    )
    return [(row.mainmsg_id, row.titlemsg_text) for row in query]


def search_titles(query_str: str, title_cache: List[tuple[int, str]], limit: int = 100) -> List[tuple[int, str, float]]:
    """Search titles using fuzzy matching and return results with scores.

    Returns:
        List of tuples: (mainmsg_id, title, score) sorted by score descending
    """
    if not title_cache:
        return []

    # Extract just the titles for fuzzy matching
    titles = [title for _, title in title_cache]

    # Use rapidfuzz to find best matches
    # scorer=fuzz.partial_ratio is good for partial matching
    matches = process.extract(
        query_str,
        titles,
        scorer=fuzz.partial_ratio,
        limit=limit
    )

    # Convert back to (mainmsg_id, title, score) format
    results = []
    for matched_title, score, idx in matches:
        mainmsg_id = title_cache[idx][0]
        results.append((mainmsg_id, matched_title, score))

    return results


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
    title_cache: List[tuple[int, str]]  # List of (mainmsg_id, title)

    def __init__(self, botargs: argparse.Namespace):
        self.botargs = botargs

        lnm_id = get_ongcode_meta("last_nonid_msg_id")
        lnm_date = get_ongcode_meta("last_nonid_msg_date")
        assert lnm_id is not None and lnm_date is not None

        self.last_nonid_msg_id = int(lnm_id)
        self.last_nonid_msg_date = datetime.datetime.fromisoformat(lnm_date)

        # Load all titles into memory for fuzzy matching
        log("INFO: Loading title database into memory...")
        self.title_cache = load_title_cache()
        log(f"INFO: Loaded {len(self.title_cache)} titles into memory")

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

        # Update in-memory cache
        cache_updated = False
        for i, (cache_id, _) in enumerate(self.title_cache):
            if cache_id == ongcode.mainmsg_id:
                # Update existing entry
                self.title_cache[i] = (ongcode.mainmsg_id, ongcode.titlemsg_text)
                cache_updated = True
                break
        
        if not cache_updated:
            # Add new entry to cache
            self.title_cache.append((ongcode.mainmsg_id, ongcode.titlemsg_text))

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


def benchmark_search(title_cache: List[tuple[int, str]]) -> None:
    """Run benchmark tests on the search functionality."""
    import random

    log("INFO: Starting search benchmark...")
    log(f"INFO: Title cache size: {len(title_cache)} entries")

    # Get a sample of titles to use as search queries
    if len(title_cache) < 100:
        sample_size = len(title_cache)
    else:
        sample_size = 100

    sample_titles = random.sample(title_cache, sample_size)

    # Test with exact matches
    log("\nINFO: Testing exact title matches...")
    start_time = time.time()
    for _, title in sample_titles:
        _ = search_titles(title, title_cache, limit=10)
    exact_time = time.time() - start_time

    log(f"INFO: Exact matches: {sample_size} searches in {exact_time:.3f}s")
    log(f"INFO: Average time per search: {(exact_time / sample_size) * 1000:.2f}ms")

    # Test with partial matches (first 3 words)
    log("\nINFO: Testing partial title matches...")
    start_time = time.time()
    for _, title in sample_titles:
        partial = ' '.join(title.split()[:3]) if len(title.split()) > 3 else title
        _ = search_titles(partial, title_cache, limit=10)
    partial_time = time.time() - start_time

    log(f"INFO: Partial matches: {sample_size} searches in {partial_time:.3f}s")
    log(f"INFO: Average time per search: {(partial_time / sample_size) * 1000:.2f}ms")

    # Test with single word searches
    log("\nINFO: Testing single word searches...")
    start_time = time.time()
    for _, title in sample_titles:
        single_word = title.split()[0] if title.split() else title
        _ = search_titles(single_word, title_cache, limit=10)
    single_time = time.time() - start_time

    log(f"INFO: Single word searches: {sample_size} searches in {single_time:.3f}s")
    log(f"INFO: Average time per search: {(single_time / sample_size) * 1000:.2f}ms")

    # Overall statistics
    total_searches = sample_size * 3
    total_time = exact_time + partial_time + single_time
    log(f"\nINFO: Overall: {total_searches} total searches in {total_time:.3f}s")
    log(f"INFO: Overall average time per search: {(total_time / total_searches) * 1000:.2f}ms")


def test_search_cli(query: str, title_cache: List[tuple[int, str]]) -> None:
    """Perform a test search from the command line without Discord."""
    log(f"INFO: Searching for: '{query}'")
    log(f"INFO: Title cache size: {len(title_cache)} entries\n")

    start_time = time.time()
    results = search_titles(query, title_cache, limit=20)
    search_time = time.time() - start_time

    log(f"INFO: Search completed in {search_time * 1000:.2f}ms")
    log(f"INFO: Found {len(results)} matches\n")

    if not results:
        print("No matches found.")
        return

    # Fetch full records for display
    matched_ids = [msg_id for msg_id, _, _ in results]
    records = {row.mainmsg_id: row for row in OngCode.select().where(OngCode.mainmsg_id.in_(matched_ids))}

    print("Top matches:")
    print("-" * 80)
    for i, (msg_id, title, score) in enumerate(results[:10], 1):
        record = records.get(msg_id)
        if record:
            # Handle both datetime objects and strings
            if isinstance(record.mainmsg_date, str):
                date_obj = datetime.datetime.fromisoformat(record.mainmsg_date)
                date_str = date_obj.strftime("%Y-%m-%d")
            else:
                date_str = record.mainmsg_date.strftime("%Y-%m-%d")
            print(f"{i:2d}. [{score:5.1f}] {date_str} - {title}")


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
        "--ongcodething-url", "--ongcodething",
        type=str,  # FIXME: Is there a "URL" type we can use?
        default=None,
        help="URL for ongcodething backend"
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

    parser.add_argument(
        "--benchmark",
        default=False,
        action="store_true",
        help="run search benchmark and exit",
    )

    parser.add_argument(
        "--test-search",
        type=str,
        default=None,
        metavar="QUERY",
        help="perform a test search without connecting to Discord",
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

    if args.debug_queries:
        import logging
        logger = logging.getLogger('peewee')
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

    if args.ongcode_guild is None:
        args.ongcode_guild = creds.get("guild")

    if args.ongcode_guild is None:
        log("ERROR: No guild specified and no guild in configuration")
        sys.exit(1)

    if args.ongcode_channel is None:
        args.ongcode_channel = creds.get("channel")

    if args.ongcode_channel is None:
        log("ERROR: No channel specified and no channel in configuration")
        sys.exit(1)

    if args.moderator_role is None:
        args.moderator_role = creds.get("moderator_role")

    if args.ongcodething_url is None:
        args.ongcodething_url = creds.get("ongcodething_url")

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

    # Handle benchmark mode
    if args.benchmark:
        log("INFO: Running in benchmark mode")
        title_cache = load_title_cache()
        log(f"INFO: Loaded {len(title_cache)} titles into memory")
        benchmark_search(title_cache)
        return

    # Handle test search mode
    if args.test_search:
        log("INFO: Running in test search mode")
        title_cache = load_title_cache()
        log(f"INFO: Loaded {len(title_cache)} titles into memory")
        test_search_cli(args.test_search, title_cache)
        return

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

        # Use fuzzy matching on in-memory title cache
        search_results = search_titles(title, bot.title_cache, limit=100)

        log(f"SEARCH RESULT COUNT: {len(search_results)}")

        # Fetch full OngCode records for the matched IDs
        if search_results:
            matched_ids = [msg_id for msg_id, _, _ in search_results]
            q = list(OngCode.select().where(OngCode.mainmsg_id.in_(matched_ids)))

            # Create a lookup dict for scores and preserve order
            score_map = {msg_id: score for msg_id, _, score in search_results}
            id_order = {msg_id: idx for idx, (msg_id, _, _) in enumerate(search_results)}

            # Sort q by the original search result order and attach scores
            q.sort(key=lambda row: id_order.get(row.mainmsg_id, 999999))
            for row in q:
                row.score = score_map.get(row.mainmsg_id, 0)
        else:
            q = []

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
                    response = f"{msg_url} - `{rowdate}` - {row.titlemsg_text} (score: {row.score:.2f})\n"
                else:
                    response = f"`{rowdate}` - {row.titlemsg_text} (score: {row.score:.2f})\n"

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

        if not discord.utils.get(ctx.author.roles, name=bot.botargs.moderator_role):
            await ctx.respond("Permission denied", ephemeral=True)
            return

        # Send initial response
        response = await ctx.respond("Sending...", ephemeral=True)
        message_obj = await response.original_message()  # FIXME: deprecated

        # Check if this message is a title or ongcode
        ongcode = OngCode.get_or_none(
            (OngCode.mainmsg_id == message.id) | (OngCode.titlemsg_id == message.id)
        )

        if ongcode is None:
            log(f"WARNING: Bad message ({message.id}) requested for ongcodething send")
            await message_obj.edit(content="This message is not a recognized ongcode or title.")
            return

        # Get the title and body
        title = ongcode.titlemsg_text or "Untitled"
        body = ongcode.mainmsg_text

        # Get the ongcodething endpoint from credentials
        ongcodething_url = bot.botargs.ongcodething_url
        if not ongcodething_url:
            log("WARNING: ongcodething called but not configured")
            await message_obj.edit(content="Error: ongcodething endpoint not configured")
            return

        # Send to codething backend
        async with aiohttp.ClientSession() as session:
            try:
                post_response = await session.post(
                    f"{ongcodething_url}/songs/",
                    json={
                        "title": title,
                        "body": body,
                        "status": "PENDING"
                    }
                )
                if post_response.status == 200:
                    log(f"INFO: Successfully sent ongcode to Jon! (id {ongcode.mainmsg_id} / Title: {title}")
                    await message_obj.edit(content=f"Successfully sent ongcode to Jon!\nTitle: {title}")
                else:
                    log(f"ERROR: Failed to send ongcode: HTTP {post_response.status}")
                    await message_obj.edit(content=f"Failed to send ongcode: HTTP {post_response.status}")
            except Exception as e:
                log(f"WARNING: Error sending ongcode to Jon! (id {ongcode.mainmsg_id} / Title: {title})")
                log(f"WARNING: {str(e)}")
                await message_obj.edit(content=f"Error sending ongcode: {str(e)}")

        # Keep existing debug printing
        # print(ppretty(ctx))
        # print(ppretty(message))
        # print(f"ZOT: {ctx.author.id}")
        # print(f"ZOT: {message.author.id}")


    # A couple of testing things
    # class MyModal(discord.ui.Modal):
    #     def __init__(self, *args, **kwargs) -> None:
    #         super().__init__(*args, **kwargs)

    #         self.add_item(discord.ui.InputText(label="Short Input"))
    #         self.add_item(discord.ui.InputText(
    #             label="Long Input", style=discord.InputTextStyle.long))

    #     async def callback(self, interaction: discord.Interaction):
    #         embed = discord.Embed(title="Modal Results")
    #         embed.add_field(name="Short Input", value=self.children[0].value)
    #         embed.add_field(name="Long Input", value=self.children[1].value)
    #         await interaction.response.send_message(embeds=[embed])

    # class MyView(discord.ui.View):
    #     @discord.ui.button(label="Send Modal")
    #     async def button_callback(self, button, interaction):
    #         await interaction.response.send_modal(MyModal(title="Modal via Button"))

    # @bot.slash_command()
    # async def send_modal(ctx):
    #     await ctx.respond(view=MyView())

    # creates a global message command. use guild_ids=[] to create guild-specific commands.
    # @bot.message_command(name="interaction_test")
    # async def interaction_test(ctx, message: discord.Message):  # message commands return the message
    #     modal = MyModal(title="Modal via Message Command")
    #     await ctx.send_modal(modal)


    bot.run(creds["token"])


if __name__ == "__main__":
    main()
