#!/usr/bin/env python
import argparse
import io
import sys
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import dateparser
import gspread
import pandas as pd
from peewee import (SQL, AutoField, CharField, DateTimeField, FloatField,
                    IntegerField, Model, SqliteDatabase)
from playhouse.sqlite_ext import (FTS5Model, RowIDField, SearchField,
                                  SqliteExtDatabase)
from tdvutil import hms_to_sec, ppretty
from tdvutil.argparse import CheckFile

# NOTE: You will need to set up a file with your google cloud credentials
# as noted in the documentation for the "gspread" module


# Where to find the onglog. Sure, we could take these as config values or
# arguments or something, but... why?
ONG_SPREADSHEET_ID = "14ARzE_zSMNhp0ZQV34ti2741PbA-5wAjsXRAW8EgJ-4"
ONG_SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{ONG_SPREADSHEET_ID}/edit"

EARLIEST_ROW=767
# EARLIEST_ROW=12900
ONG_RANGE_NAME = f"Songs!A{EARLIEST_ROW}:I"


class OngLog(Model):
    # __tablename__ = 'onglog'
    rowid = IntegerField(primary_key=True, null=False)
    start_time = DateTimeField(index=True, null=False)
    end_time = DateTimeField(index=True, null=False)
    stream_uptime = FloatField(null=True)
    requester = CharField(null=True, index=True)
    title = CharField(null=False)
    genre = CharField(null=True)
    request_type = CharField(null=True)
    looper_slot = CharField(null=True)
    looper_file_number = IntegerField(null=True)


class OngLogIndex(FTS5Model):
    rowid = RowIDField()
    title = SearchField()

    class Meta:
        options = {"tokenize": "unicode61"}


class OngLogMeta(Model):
    key = CharField(primary_key=True)
    value = CharField()


_db: SqliteExtDatabase
_db_initialized = False
def initialize_db():
    global _db_initialized
    if _db_initialized:
        return

    onglog_db_file = Path(__file__).parent / 'onglog.db'

    global _db
    _db = SqliteExtDatabase(None)
    _db.init(onglog_db_file, pragmas={"journal_mode": "wal", "cache_size": -1 * 64 * 1024})
    _db.bind([OngLog, OngLogMeta, OngLogIndex])
    _db.connect()
    _db.create_tables([OngLog, OngLogMeta, OngLogIndex])

    _db_initialized = True


def get_db():
    initialize_db()

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


# # Create a datetime object for Saturday, September 18, 2024 at 8:00 AM
# target_datetime = datetime(2024, 9, 29, 2, 0)
# print(f"Target datetime: {target_datetime}")

# # Find onglog entries for the target datetime
# entries = find_onglog_entry_by_datetime(target_datetime)
# print(f"Onglog entries found: {ppretty(entries)}")


def log(msg):
    print(msg)
    sys.stdout.flush()


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a video file based on timestamps in the onglog",
        allow_abbrev=True,
    )

    parser.add_argument(
        "--credsfile",
        default="gsheets_credentials.json",
        type=Path,
        action=CheckFile(must_exist=True),
        help="credentials file to use",
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

    parsed_args = parser.parse_args(argv)
    return parsed_args


# This is really a mess. Can we do better?
def main(argv: List[str]) -> int:
    args = parse_arguments(argv[1:])
    onglog_tmp = Path(f"onglog_tmp.xlsx")
    gc = gspread.service_account(filename=args.credsfile)
    # onglog = gc.open_by_url(ONG_SPREADSHEET_URL)
    # ws = onglog.worksheet("Songs")

    initialize_db()

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
    df = pd.read_excel(onglog_tmp, header=None, sheet_name="Songs", names=["Date", "Uptime", "Order", "Requester", "Title", "Genre", "Type", "Links", "Looper", "FileNum"])

    last_processed_row = get_onglog_meta("last_processed_row")

    resume_row = 0
    if last_processed_row:
        start_row_num = int(last_processed_row) - 200 - 1
        resume_row = int(last_processed_row)
        log(f"Resuming from row {resume_row}")
    else:
        log(f"Starting from row {EARLIEST_ROW}")
        start_row_num = EARLIEST_ROW

    # Find the first row with 'Order' value of 0 after or including EARLIEST_ROW
    order_zero_rows = df[df['Order'] == 0]
    valid_rows = order_zero_rows[order_zero_rows.index >= start_row_num - 1]
    start_index = valid_rows.index.min() if not valid_rows.empty else None

    if not start_index:
        log(f"No rows with 'Order' value of 0 found after row {start_row_num}. Using EARLIEST_ROW as starting point.")
        start_index = EARLIEST_ROW - 1
    else:
        log(f"Starting processing from row {start_index + 1}")


    eastern = ZoneInfo("America/New_York")
    sydney = ZoneInfo("Australia/Sydney")

    db = get_db()

    # Adjust the loop to start from the identified row
    df_subset = df.iloc[start_index:]

    with db.atomic():
        for index, row in df_subset.iterrows():
            assert isinstance(index, int)

            if index >= resume_row:
                log(f"Processing row {index + 1}: {row.Title} ({row.Order})")

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
                requester=row['Requester'] if pd.notna(row['Requester']) else None,
                title=row['Title'],
                genre=row['Genre'] if pd.notna(row['Genre']) else None,
                request_type=row['Type'] if pd.notna(row['Type']) else None,
                looper_slot=row['Looper'] if pd.notna(row['Looper']) else None,
                looper_file_number=row['FileNum'] if pd.notna(row['FileNum']) else None
            )
            onglog_entry.execute()

            onglog_index_entry = OngLogIndex.replace(
                rowid=index + 1,
                title=row['Title']
            )
            onglog_index_entry.execute()


    set_onglog_meta("last_processed_row", str(index + 1))

    return 0


if __name__ == "__main__":
    # make sure our output streams are properly encoded so that we can
    # not screw up Frédéric Chopin's name and such.
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8")

    main(sys.argv)
else:
    initialize_db()
