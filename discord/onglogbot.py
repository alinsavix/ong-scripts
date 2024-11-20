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
from discord.ext import pages

# NOTE: You will need to set up a file with your google cloud credentials
# as noted in the documentation for the "gspread" module


# Where to find the onglog. Sure, we could take these as config values or
# arguments or something, but... why?
ONG_SPREADSHEET_ID = "14ARzE_zSMNhp0ZQV34ti2741PbA-5wAjsXRAW8EgJ-4"
ONG_SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{ONG_SPREADSHEET_ID}/edit"

EARLIEST_ROW=767
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
    _db.init(dbfile, pragmas={"journal_mode": "wal", "cache_size": -1 * 64 * 1024, "foreign_keys": 1})
    _db.bind([OngLogTitle, OngLogIndex, OngLog, OngLogMeta])
    _db.connect()
    _db.create_tables([OngLogTitle, OngLogIndex, OngLog, OngLogMeta])

    _db_initialized = True


def get_db():
    global _db
    return _db


# Example function to query onglog entries
# def get_onglog_entries():
#     get_db()
#     return session.query(OngLog).all()


# Function to find onglog entry by a given date & time
def find_onglog_entry_by_datetime(datetime):
    # db = get_db()
    # return session.query(OngLog).filter(OngLog.start_time <= datetime, OngLog.end_time >= datetime).first()
    return OngLog.select().where((OngLog.start_time <= datetime) & (OngLog.end_time >= datetime)).first()


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

# # Create a datetime object for Saturday, September 18, 2024 at 8:00 AM
# target_datetime = datetime(2024, 9, 29, 2, 0)
# print(f"Target datetime: {target_datetime}")

# # Find onglog entries for the target datetime
# entries = find_onglog_entry_by_datetime(target_datetime)
# print(f"Onglog entries found: {ppretty(entries)}")


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
    onglog_tmp = Path(f"onglog_{args.environment}.xlsx")
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
        log(f"INFO: No rows with 'Order' value of 0 found after row {
            start_row_num}. Using EARLIEST_ROW as starting point.")
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
                requester=row['Requester'] if (pd.notna(row['Requester']) and row['Requester'] != "-") else None,
                titleid=get_title_id(row['Title']),
                genre=row['Genre'] if pd.notna(row['Genre']) else None,
                request_type=row['Type'] if pd.notna(row['Type']) else None,
                notes=str(row['Links']) if (pd.notna(row['Links']) and "Highlight" not in str(row['Links'])) else None,
                looper_slot=row['Looper'] if pd.notna(row['Looper']) else None,
                looper_file_number=row['FileNum'] if pd.notna(row['FileNum']) else None
            )
            onglog_entry.execute()

            # onglog_index_entry = OngLogIndex.replace(
            #     rowid=index + 1,
            #     title=row['Title']
            # )
            # onglog_index_entry.execute()

    set_onglog_meta("last_processed_row", str(index + 1))


bot_guild = None

# make sure to also read https://guide.pycord.dev/getting-started/more-features
class OnglogBot(discord.Bot):
    botchannel: discord.TextChannel
    # botguild: discord.Guild

    def __init__(self, botargs: argparse.Namespace):
        self.botargs = botargs

        # lnm_id = get_onglog_meta("last_nonid_msg_id")
        # lnm_date = get_onglog_meta("last_nonid_msg_date")
        # assert lnm_id is not None and lnm_date is not None

        # self.last_nonid_msg_id = int(lnm_id)
        # self.last_nonid_msg_date = datetime.fromisoformat(lnm_date)

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

        # log(f"finding channel #{self.botargs.onglog_channel}")
        # channel = discord.utils.get(self.get_all_channels(
        # ), guild__name=self.botargs.onglog_guild, name=self.botargs.onglog_channel)

        # if channel is None:
        #     log(f"ERROR: channel #{self.botargs.onglog_channel} not found, can't reasonably continue")
        #     os._exit(1)

        # log(f"found channel with id {channel.id}")
        # self.botchannel = channel
        # global bot_channel, bot_guild
        # bot_channel = channel
        # bot_guild = channel.guild

        # await self.message_catchup()
        # sys.stdout.flush()





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
        "--onglog-guild",
        type=str,
        default=None,
        help="Discord guild (server) to use for onglog things"
    )

    parser.add_argument(
        "--dbfile",
        type=Path,
        default=None,
        help="database file to use"
    )

    # FIXME: make this work again
    # parser.add_argument(
    #     "--onglog-db-file",
    #     type=Path,
    #     default=Path(__file__).parent / 'onglog.db',
    #     help="location of onglog database",
    # )

    parser.add_argument(
        "--force-fetch",
        default=False,
        action="store_true",
        help="force fetching the onglog, even if ours is recent",
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
        parsed_args.gsheets_credentials_file = Path(__file__).parent / "gsheets_credentials.json"

    if parsed_args.dbfile is None:
        parsed_args.dbfile = Path(__file__).parent / f"onglog_{parsed_args.environment}.db"

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

    if args.onglog_guild is None:
        args.onglog_guild = creds.get("guild", None)

    if args.onglog_guild is None:
        log("ERROR: No guild specified and no guild in configuration")
        sys.exit(1)

    # if args.ongcode_channel is None:
    #     args.ongcode_channel = creds.get("channel", None)

    # if args.ongcode_channel is None:
    #     log("ERROR: No channel specified and no channel in configuration")
    #     sys.exit(1)

    # if args.moderator_role is None:
    #     args.moderator_role = creds.get("moderator_role", None)

    log("INFO: In startup")
    log(f"INFO: Using guild '{args.onglog_guild}'")
    # log(f"INFO: Using channel '{args.ongcode_channel}'")

    initialize_db(args.dbfile)
    onglog_update(args)

    log("INFO: onglog processing complete")

    bot = OnglogBot(botargs=args)

    @bot.slash_command(name="onglog", description="Search for ongcode")
    async def cmd_find_onglog(
        ctx: discord.ApplicationContext,
        title: discord.Option(str, "Partial song title"),
    ):
        # await ctx.trigger_typing()
        # await asyncio.sleep(1)
        log(f"SEARCH: '{title}'")

        # x = (
        #     OngLog
        #     .select(OngLog, OngLogIndex.rank().alias("score"), OngLogTitle)
        #     .join(OngLogTitle, on=(OngLog.titleid == OngLogTitle.rowid))
        #     .join(OngLogIndex, on=(OngLog.rowid == OngLogIndex.rowid))
        #     .where(OngLogIndex.match(title))
        #     .order_by(OngLogIndex.rank())
        # )

        x = (
            OngLogIndex
            .select(OngLog.titleid, OngLogIndex.title, OngLogIndex.rank().alias("score"), fn.DATE(fn.MAX(OngLog.start_time)).alias("last_played"))
            .join(OngLog, on=(OngLogIndex.rowid == OngLog.titleid))
            .where(OngLogIndex.match(title))
            .order_by(OngLogIndex.rank())
            .group_by(OngLog.titleid)
        )

        log(f"SEARCH RESULT COUNT: {len(x)}")

        if len(x) == 0:
            embed = discord.Embed(
                title="Ongcode Search",
                # description="I'm the Ongcode bot. I'm here to help you find ongcode in the channel",
                color=discord.Color.red()
            )
            embed.add_field(name="", value="No matches found", inline=False)
            await ctx.respond(embed=embed, ephemeral=True)
            return

        pagelist = []

        for i, chunk in enumerate(ichunked(x, MATCH_LIMIT)):
            response_all = f"### Ongcode Search Results (page {i + 1} of {(len(x) // MATCH_LIMIT) + 1})\n"
            # response_all += "**"
            for row in chunk:
                print(ppretty(row))
                # rowdate = datetime.fromisoformat(str(row.start_time)).date()

                # msg_url = f"{ONG_SPREADSHEET_URL}?range=A{row.rowid}"
                response = f"*`{row.onglog.titleid}`* - `{row.last_played}` - {row.title} (score: {abs(row.score):.2f})\n"

                response_all += response

            pagelist.append(
                pages.Page(content=response_all)
            )

        paginator = pages.Paginator(pages=pagelist, disable_on_timeout=True, timeout=600)
        await paginator.respond(ctx.interaction, ephemeral=True)

        sys.stdout.flush()

    bot.run(creds["token"])
    return 0
    # onglog_tmp = Path(f"onglog_{args.environment}.xlsx")
    # gc = gspread.service_account(filename=args.gsheets_credentials_file)
    # onglog = gc.open_by_url(ONG_SPREADSHEET_URL)
    # ws = onglog.worksheet("Songs")

    # a bunch of testing code, commented out, that we need to come back to
    # z = datetime(2024,10,6,5,40,15)
    # print(z)
    # x = find_onglog_entry_by_datetime(z)
    # print(x)
    # sys.exit(0)

    # x = OngLog.select().join(OngLogIndex, on=(OngLog.rowid == OngLogIndex.rowid)).where(OngLogIndex.match("weight of the world"))

    # x = OngLogIndex.search(
    #     "weight world",
    #     weights={"title": 2.0},
    #     with_score=True,
    #     score_alias="score").join(OngLog, on=(OngLogIndex.rowid == OngLog.rowid)).distinct([OngLog.title]).execute()

    # x = OngLogIndex.search(
    #     "weight world",
    #     weights={"title": 2.0},
    #     with_score=True,
    #     score_alias="score").order_by(SQL("score")).execute() # order_by(OngLogIndex.score.desc()).limit(10).execute()

    # res = OngLogIndex.search(
    #     "weight world",
    #     weights={"title": 2.0},
    #     with_score=True,
    #     score_alias="score").execute() # order_by(OngLogIndex.score.desc()).limit(10).execute()

    # things = {x.title: x.score for x in res}
    # print(things)
    # subq = OngLogIndex.search(
    #     "weight world",
    #     weights={"title": 2.0},
    #     with_score=True,
    #     score_alias="score").alias("subq")

    # x = OngLog.select(subq.c.rowid, subq.c.score, subq.c.title).execute()
    # for row in x:
    #     print(row.rowid, row.score, row.title)




if __name__ == "__main__":
    main()
