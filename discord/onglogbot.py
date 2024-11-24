#!/usr/bin/env python
import argparse
import io
import os
import sys
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import dateparser
import gspread
import pandas as pd
import toml
from more_itertools import ichunked
from peewee import (SQL, AutoField, CharField, DateTimeField, FloatField,
                    ForeignKeyField, IntegerField, Model, SqliteDatabase, fn)
from playhouse.sqlite_ext import (FTS5Model, RowIDField, SearchField,
                                  SqliteExtDatabase)
from tdvutil import hms_to_sec, ppretty
from tdvutil.argparse import CheckFile

import discord
from discord.commands import SlashCommandGroup
from discord.ext import commands, pages

# NOTE: You will need to set up a file with your google cloud credentials
# as noted in the documentation for the "gspread" module


# Where to find the onglog. Sure, we could take these as config values or
# arguments or something, but... why?
ONG_SPREADSHEET_ID = "14ARzE_zSMNhp0ZQV34ti2741PbA-5wAjsXRAW8EgJ-4"
ONG_SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{ONG_SPREADSHEET_ID}/edit"

EARLIEST_ROW = 767
# EARLIEST_ROW=12900
ONG_RANGE_NAME = f"Songs!A{EARLIEST_ROW}:I"

MATCH_LIMIT = 10



class OngLogTitle(Model):
    rowid = AutoField()
    title = CharField(null=False, index=True)

    class Meta:
        table_name = "onglog_title"

class OngLogIndex(FTS5Model):
    rowid = RowIDField()
    title = SearchField()

    class Meta:
        table_name = "onglog_index"
        options = {"tokenize": "unicode61"}

class OngLog(Model):
    # __tablename__ = 'onglog'
    rowid = IntegerField(primary_key=True, null=False)
    start_time = DateTimeField(index=True, null=False)
    end_time = DateTimeField(index=True, null=False)
    stream_uptime = FloatField(null=True)
    requester = CharField(null=True, index=True)
    titleid = IntegerField(null=False, index=True)
    genre = CharField(null=True)
    request_type = CharField(null=True)
    notes = CharField(null=True)
    looper_slot = CharField(null=True)
    looper_file_number = IntegerField(null=True)

    class Meta:
        table_name = "onglog"

class OngLogMeta(Model):
    key = CharField(primary_key=True)
    value = CharField()


_db: SqliteExtDatabase
_db_initialized = False
def initialize_db(dbfile: Path):
    global _db_initialized
    if _db_initialized:
        return

    # onglog_db_file = Path(__file__).parent / 'onglog.db'
    log(f"INFO: Using database file {dbfile}")

    global _db
    _db = SqliteExtDatabase(None)
    _db.init(dbfile, pragmas={"journal_mode": "wal",
             "cache_size": -1 * 64 * 1024, "foreign_keys": 1})
    _db.bind([OngLogTitle, OngLogIndex, OngLog, OngLogMeta])
    _db.connect()
    _db.create_tables([OngLogTitle, OngLogIndex, OngLog, OngLogMeta])

    _db_initialized = True


def get_db():
    global _db
    return _db



def set_onglog_meta(key: str, value: str):
    meta = OngLogMeta.replace(
        key=key,
        value=value
    )
    meta.execute()


def get_onglog_meta(key: str) -> Optional[str]:
    m = OngLogMeta.get_or_none(OngLogMeta.key == key)
    if m:
        return m.value

    # else
    return None


def get_title_id(title: str) -> int:
    t = OngLogTitle.get_or_none(OngLogTitle.title == title)
    if t:
        return t.rowid

    # else, gotta add it
    t = OngLogTitle.create(title=title)
    idx = OngLogIndex.replace(
        rowid=t.rowid,
        title=title
    )
    idx.execute()

    return t.rowid


def log(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.stdout.flush()

def get_credentials(cfgfile: Path, environment: str) -> Dict[str, str]:
    log(f"loading config from {cfgfile}")
    config = toml.load(cfgfile)

    try:
        return config["onglog_bot"][environment]
    except KeyError:
        log(f"ERROR: no configuration for onglog_bot.{environment} in credentials file")
        sys.exit(1)


def onglog_update(args: argparse.Namespace):
    onglog_tmp = Path(__file__).parent / f"onglog_{args.environment}.xlsx"
    gc = gspread.service_account(filename=args.gsheets_credentials_file)

    # Check if the onglog_tmp file is older than 8 hours
    if onglog_tmp.exists():
        file_age = datetime.now() - datetime.fromtimestamp(onglog_tmp.stat().st_mtime)
        log(f"INFO: The onglog_tmp file is {file_age.total_seconds() / 3600:.1f} hours old.")

        if file_age > timedelta(hours=8) or args.force_fetch:
            log("INFO: Fetching a fresh copy of the onglog...")
            dl = gc.export(file_id=ONG_SPREADSHEET_ID, format=gspread.utils.ExportFormat.EXCEL)
            onglog_tmp.write_bytes(dl)
            log(f"INFO: Saved onglog as {onglog_tmp}")
        else:
            log(f"INFO: Using existing onglog copy at {onglog_tmp}")
    else:
        log("INFO: Fetching an initial copy of the onglog...")
        dl = gc.export(file_id=ONG_SPREADSHEET_ID, format=gspread.utils.ExportFormat.EXCEL)
        onglog_tmp.write_bytes(dl)
        log(f"INFO: Saved onglog as {onglog_tmp}")


    # By saying no header, it means we can keep the onglong row number equal
    # to the pd row number + 1
    df = pd.read_excel(onglog_tmp, header=None, sheet_name="Songs", names=[
                       "Date", "Uptime", "Order", "Requester", "Title", "Genre", "Type", "Links", "Looper", "FileNum"])

    last_processed_row = get_onglog_meta("last_processed_row")

    resume_row = 0
    if last_processed_row:
        start_row_num = int(last_processed_row) - 200 - 1
        resume_row = int(last_processed_row)
        log(f"INFO: Resuming onglog processing from row {resume_row}")
    else:
        log(f"INFO: Starting onglog processing from row {EARLIEST_ROW}")
        start_row_num = EARLIEST_ROW

    # Find the first row with 'Order' value of 0 after or including EARLIEST_ROW
    order_zero_rows = df[df['Order'] == 0]
    valid_rows = order_zero_rows[order_zero_rows.index >= start_row_num - 1]
    start_index = valid_rows.index.min() if not valid_rows.empty else None

    if not start_index:
        log(f"INFO: No rows with 'Order' value of 0 found after row {start_row_num}. Using EARLIEST_ROW as starting point.")
        start_index = EARLIEST_ROW - 1
    else:
        log(f"INFO: Starting processing from row {start_index + 1}")


    eastern = ZoneInfo("America/New_York")
    sydney = ZoneInfo("Australia/Sydney")

    db = get_db()

    # Adjust the loop to start from the identified row
    df_subset = df.iloc[start_index:]

    with db.atomic():
        for index, row in df_subset.iterrows():
            assert isinstance(index, int)

            if index >= resume_row:
                log(f"INFO: Processing row {index + 1}: {row.Title} ({row.Order})")

            if not pd.notna(row['Date']):
                continue

            if row['Order'] == "-" or int(row['Order']) == 0:
                continue

            if not row['Date'] or row['Date'] == "-":
                continue

            # Convert start_time to Sydney time
            start_time_eastern = dateparser.parse(str(row['Date']))
            if start_time_eastern:
                start_time_eastern = start_time_eastern.replace(tzinfo=eastern)
                start_time = start_time_eastern.astimezone(sydney)
            else:
                start_time = None

            # Determine end_time based on the next row
            if index + 1 < len(df):
                next_row = df.iloc[index + 1]
                if next_row['Order'] == 0:  # Start of a new stream
                    end_time = start_time + timedelta(hours=2) if start_time else None
                elif next_row['Date'] == "-":  # hopefully rare corner case
                    continue
                else:
                    end_time_eastern = dateparser.parse(str(next_row['Date']))
                    if end_time_eastern:
                        end_time_eastern = end_time_eastern.replace(tzinfo=eastern)
                        end_time = end_time_eastern.astimezone(sydney)
                    else:
                        end_time = None
            else:  # Last row in the dataframe
                end_time = start_time + timedelta(hours=2) if start_time else None

            try:
                uptime = hms_to_sec(str(row['Uptime']))
            except ValueError:
                uptime = None

            onglog_entry = OngLog.replace(
                rowid=index + 1,
                start_time=start_time,
                end_time=end_time,
                stream_uptime=uptime,
                requester=row['Requester'] if (
                    pd.notna(row['Requester']) and row['Requester'] != "-") else None,
                titleid=get_title_id(row['Title']),
                genre=row['Genre'] if pd.notna(row['Genre']) else None,
                request_type=row['Type'] if pd.notna(row['Type']) else None,
                notes=str(row['Links']) if (pd.notna(row['Links'])
                                            and "Highlight" not in str(row['Links'])) else None,
                looper_slot=row['Looper'] if pd.notna(row['Looper']) else None,
                looper_file_number=row['FileNum'] if pd.notna(row['FileNum']) else None
            )
            onglog_entry.execute()


    set_onglog_meta("last_processed_row", str(index + 1))


class OnglogBot(discord.Bot):
    def __init__(self, botargs: argparse.Namespace):
        self.botargs = botargs

        intents = discord.Intents.default()
        # intents.presences = True
        # intents.messages = True
        # intents.message_content = True
        intents.reactions = True
        # intents.typing = True
        # intents.members = True

        super().__init__(intents=intents)   # , status=discord.Status.invisible)

    async def on_ready(self):
        # print(ppretty(self))
        log(f"{self.user} (id {self.user.id}) is online")


class OnglogCommands(commands.Cog):
    def __init__(self, bot: discord.Bot):
        self.bot = bot

    onglog_cmds = SlashCommandGroup("onglog", "onglog search & info commands")
    title_cmds = onglog_cmds.create_subgroup("title", "song title commands")
    user_cmds = onglog_cmds.create_subgroup("user", "user commands")

    @title_cmds.command(name="search", description="Search onglog for song title")
    async def cmd_onglog_title_search(
        self,
        ctx: discord.ApplicationContext,
        title: discord.Option(str, "Partial song title"),
    ):
        # await ctx.trigger_typing()
        # await asyncio.sleep(1)
        log(f"SEARCH: '{title}'")

        title = title.replace(",", " ")

        q = (
            OngLogIndex
            .select(OngLog.titleid, OngLogIndex.title, OngLogIndex.rank().alias("score"),
                    fn.DATE(fn.MAX(OngLog.start_time)).alias("last_played"))
            .join(OngLog, on=(OngLogIndex.rowid == OngLog.titleid))
            .where(OngLogIndex.match(title))
            .order_by(OngLogIndex.rank())
            .group_by(OngLog.titleid)
        )

        log(f"SEARCH RESULT COUNT: {len(q)}")

        if len(q) == 0:
            embed = discord.Embed(
                title="Onglog Title Search",
                color=discord.Color.red()
            )
            embed.add_field(name="", value="No matches found", inline=False)
            await ctx.respond(embed=embed, ephemeral=True)
            return

        pagelist = []

        page_count = (len(q) // MATCH_LIMIT) + 1
        for i, chunk in enumerate(ichunked(q, MATCH_LIMIT)):
            page_num = i + 1
            response_all = f"### Onglog Title Search Results (page {page_num} of {page_count})\n"
            response_all += f"`{"id":>5}` - `last play ` - `title`\n"
            for row in chunk:
                # print(ppretty(row))

                # msg_url = f"{ONG_SPREADSHEET_URL}?range=A{row.rowid}"
                # (score: {abs(row.score):.2f})\n"
                response = f"*`{row.onglog.titleid:>5}`* - `{row.last_played}` - {row.title}\n"

                response_all += response

            if page_num == 1:
                response_all += "\nUse `/onglog title info <id>` for individual song info"

            pagelist.append(
                pages.Page(content=response_all)
            )

        paginator = pages.Paginator(pages=pagelist, disable_on_timeout=True, show_disabled=False, timeout=15)
        await paginator.respond(ctx.interaction, ephemeral=True)

        sys.stdout.flush()


    @title_cmds.command(name="info", description="Give info/stats for a song from the onglog")
    async def cmd_onglog_title_info(
        self,
        ctx: discord.ApplicationContext,
        titleid: discord.Option(int, "Title id"),
    ):
        # await ctx.trigger_typing()
        # await asyncio.sleep(1)
        log(f"SEARCH: id {titleid}")

        # Make sure the titleid requested actually exists.
        q = OngLogTitle.get_or_none(OngLogTitle.rowid == titleid)
        if q is None:
            embed = discord.Embed(
                title="Onglog Title Info",
                color=discord.Color.red()
            )
            embed.add_field(name="", value=f"There is no song title with the id '{titleid}'", inline=False)
            await ctx.respond(embed=embed, ephemeral=True)
            return

        title = q.title
        print(f"TITLE: {title}")

        # Okay, so it exists (and we know the title), so lets generate some stats.
        q = (
            OngLog
            .select(OngLog.request_type, fn.COUNT().alias("play_count"))
            .where(OngLog.titleid == titleid)
            .group_by(OngLog.request_type)
        )

        play_count = {row.request_type: row.play_count for row in q}
        play_total = sum(play_count.values())
        play_count["Loop"] = 0 if "Loop" not in play_count else play_count["Loop"]
        play_count["Piano"] = 0 if "Piano" not in play_count else play_count["Piano"]
        play_other = play_total - play_count['Loop'] - play_count['Piano']

        response_all = f"### {title}\nTitle id {titleid}\n"
        response_all += f"Played {play_total} times ({play_count['Loop']} loops, {play_count['Piano']} piano-only"
        if play_other > 0:
            response_all += f", {play_other} other"
        response_all += ")\n\n"

        q = (
            OngLog
            .select(OngLog.requester, OngLog.start_time, fn.DATE(OngLog.start_time).alias("start_date"))
            .where(OngLog.titleid == titleid)
            .order_by(OngLog.start_time.desc())
            .limit(1)
        )

        last_played = q[0].start_date.date()
        last_req = q[0].requester

        q = (
            OngLog
            .select(OngLog.requester, OngLog.start_time, fn.DATE(OngLog.start_time).alias("start_date"))
            .where(OngLog.titleid == titleid)
            .order_by(OngLog.start_time.asc())
            .limit(1)
        )

        first_played = q[0].start_date.date()
        first_req = q[0].requester


        if first_played == last_played:
            response_all += f"**Played:** {first_played}"
            if first_req:
                response_all += f" (req'd by {first_req})"
        else:
            response_all += f"**First played:** {first_played}"
            if first_req:
                response_all += f" (req'd by {first_req})"
            response_all += "\n"

            response_all += f"**Last played:** {last_played}"
            if last_req:
                response_all += f"(req'd by {last_req})"
            response_all += "\n"

        # embed = discord.Embed(
        #     title="Onglog Search",
        #     # description="I'm the Onglog bot. I'm here to help you search the onglog",
        #     color=discord.Color.red()
        # )
        # embed.add_field(name="", value=response_all)
        await ctx.respond(response_all, ephemeral=True)
        return


    @user_cmds.command(name="info", description="Give info/stats for a twitch user")
    async def cmd_onglog_user_info(
        self,
        ctx: discord.ApplicationContext,
        username: discord.Option(str, "Twitch username"),
    ):
        # await ctx.trigger_typing()
        # await asyncio.sleep(1)
        log(f"SEARCH: user {username}")

        q = (
            OngLog
            .select(OngLog.rowid.alias("onglog_line"), fn.DATE(OngLog.start_time).alias("play_date"),
                    OngLogTitle.title, OngLog.request_type)
            .join(OngLogTitle, on=(OngLog.titleid == OngLogTitle.rowid))
            .where(
                (OngLog.requester == username) &
                (
                    OngLog.notes.is_null() |
                    ~(OngLog.notes.contains("tier"))
                )
            )
            .order_by(OngLog.start_time.asc())
        )

        if len(q) == 0:
            embed = discord.Embed(
                title="Onglog User Info",
                color=discord.Color.red()
            )
            embed.add_field(name="", value=f"No requests found for user '{username}'", inline=False)
            await ctx.respond(embed=embed, ephemeral=True)
            return

        # for row in q:
        #     print(ppretty(row))

        # Count occurrences of each request type
        from collections import Counter
        req_counts = Counter(row.request_type for row in q)

        req_total = sum(req_counts.values())
        req_counts["Loop"] = 0 if "Loop" not in req_counts else req_counts["Loop"]
        req_counts["Piano"] = 0 if "Piano" not in req_counts else req_counts["Piano"]
        req_other = req_total - req_counts['Loop'] - req_counts['Piano']

        # generate the response
        response_all = f"### Twitch user '{username}'\n\n"
        response_all += f"{req_total} total requests ({req_counts['Loop']} loops, {req_counts['Piano']} piano-only"
        if req_other > 0:
            response_all += f", {req_other} other"
        response_all += ")\n\n"

        first_req_date = q[0].play_date.date()
        response_all += "**First request:**\n"
        response_all += f"`{first_req_date}` - {q[0].onglogtitle.title}\n\n"

        response_all += "**Most recent requests:**\n"
        for row in q[-5:]:
            req_date = row.play_date.date()
            response_all += f"`{req_date}` - {row.onglogtitle.title}\n"

        # embed = discord.Embed(
        #     title="Ongcode Search",
        #     # description="I'm the Ongcode bot. I'm here to help you find ongcode in the channel",
        #     color=discord.Color.red()
        # )
        # embed.add_field(name="", value=response_all)
        await ctx.respond(response_all, ephemeral=True)
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the onglog via discord",
    )

    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=None,
        action=CheckFile(must_exist=True),
        help="file with discord credentials"
    )

    parser.add_argument(
        "--gsheets-credentials-file",
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
        "--dbfile",
        type=Path,
        default=None,
        help="database file to use"
    )

    parser.add_argument(
        "--force-fetch",
        default=False,
        action="store_true",
        help="force fetching the onglog, even if ours is recent",
    )

    parser.add_argument(
        "--update-only",
        default=False,
        action="store_true",
        help="update the onglog data and then exit",
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

    if parsed_args.gsheets_credentials_file is None:
        parsed_args.gsheets_credentials_file = Path(
            __file__).parent / "gsheets_credentials.json"

    if parsed_args.dbfile is None:
        parsed_args.dbfile = Path(__file__).parent / f"onglog_{parsed_args.environment}.db"

    if parsed_args.update_only:
        parsed_args.force_fetch = True

    return parsed_args


# This is really a mess. Can we do better?
def main() -> int:
    # make sure our output streams are properly encoded so that we can
    # not screw up Frédéric Chopin's name and such.
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8", line_buffering=True)

    args = parse_args()

    if args.debug_queries:
        import logging
        logger = logging.getLogger('peewee')
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

    creds = get_credentials(args.credentials_file, args.environment)

    log("INFO: In startup")

    initialize_db(args.dbfile)
    onglog_update(args)

    log("INFO: onglog processing complete")

    if args.update_only:
        log("INFO: update-only mode, exiting")
        return 0

    bot = OnglogBot(botargs=args)
    bot.add_cog(OnglogCommands(bot))

    bot.run(creds["token"])
    return 0


if __name__ == "__main__":
    main()
