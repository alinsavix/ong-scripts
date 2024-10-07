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
import sqlalchemy
from sqlalchemy import (Column, DateTime, Float, Integer, String, Time,
                        create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker
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


Base = declarative_base()

class OngLog(Base):
    __tablename__ = 'onglog'

    id = Column(Integer, primary_key=True, autoincrement=True)
    onglog_line_number = Column(sqlalchemy.Integer, unique=True, nullable=False, index=True)
    start_time = Column(sqlalchemy.DateTime, nullable=False, index=True)
    end_time = Column(sqlalchemy.DateTime, nullable=False, index=True)
    stream_uptime = Column(sqlalchemy.Float, nullable=True)
    requester = Column(sqlalchemy.String, nullable=True)
    title = Column(sqlalchemy.String, nullable=False)
    genre = Column(sqlalchemy.String, nullable=True)
    request_type = Column(sqlalchemy.String, nullable=True)
    looper_slot = Column(sqlalchemy.Integer, nullable=True)
    looper_file_number = Column(sqlalchemy.Integer, nullable=True)


class OngLogMeta(Base):
    __tablename__ = 'onglogmeta'

    key = Column(String, nullable=False, primary_key=True)
    value = Column(String, nullable=False)


_session_initialized = False
def init_db_session():
    global _session_initialized
    if _session_initialized:
        return

    # Database setup
    onglog_db_file = Path(__file__).parent / 'onglog.db'
    engine = sqlalchemy.create_engine(f"sqlite:///{onglog_db_file}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    global session
    session = Session()

    _session_initialized = True


# Example function to query onglog entries
def get_onglog_entries():
    init_db_session()
    return session.query(OngLog).all()


# Function to find onglog entry by a given date & time
def find_onglog_entry_by_datetime(datetime):
    init_db_session()
    return session.query(OngLog).filter(OngLog.start_time <= datetime, OngLog.end_time >= datetime).first()


def set_onglog_meta(key: str, value: str):
    existing_entry = session.query(OngLogMeta).filter_by(key=key).first()

    if existing_entry:
        existing_entry.value = value
    else:
        new_entry = OngLogMeta(key=key, value=value)
        session.add(new_entry)

    try:
        session.commit()
    except sqlalchemy.exc.SQLAlchemyError as e:
        session.rollback()
        log(f"Error setting OngLogMeta: {e}")


def get_onglog_meta(key: str) -> Optional[str]:
    entry = session.query(OngLogMeta).filter_by(key=key).first()
    if entry:
        return entry.value

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

    parser.add_argument(
        "--onglog-db-file",
        type=Path,
        default=Path(__file__).parent / 'onglog.db',
        help="location of onglog database",
    )

    parser.add_argument(
        "--skip-fetch",
        default=False,
        action="store_true",
        help="skip fetching the onglog",
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

    if not args.skip_fetch:
        log("Fetching onglog...")
        dl = gc.export(file_id=ONG_SPREADSHEET_ID, format=gspread.utils.ExportFormat.EXCEL)
        onglog_tmp.write_bytes(dl)
        log(f"Saved onglog backup as {onglog_tmp}")
    else:
        log(f"Using existing onglog backup at {onglog_tmp}")

    init_db_session()

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

    # Adjust the loop to start from the identified row
    df_subset = df.iloc[start_index:]
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

        onglog_entry = OngLog(
            onglog_line_number=index + 1,
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

        try:
            session.add(onglog_entry)
            session.commit()
        except sqlalchemy.exc.IntegrityError:
            session.rollback()
            existing_entry = session.query(OngLog).filter_by(onglog_line_number=onglog_entry.onglog_line_number).first()
            if existing_entry:
                session.delete(existing_entry)
                session.flush()
            session.add(onglog_entry)
            session.commit()

    set_onglog_meta("last_processed_row", str(index + 1))

    return 0


if __name__ == "__main__":
    # make sure our output streams are properly encoded so that we can
    # not screw up Frédéric Chopin's name and such.
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8")

    main(sys.argv)

#     import pandas as pd

#     def read_xlsx_and_insert_to_db(xlsx_file):
#         # Read the xlsx file
#         df = pd.read_excel(xlsx_file)

#         # Iterate over the rows in the dataframe and insert them into the database
#         for index, row in df.iterrows():
#             add_onglog_entry(
#                 onglog_line_number=row['onglog_line_number'],
#                 start_time=row['start_time'],
#                 end_time=row['end_time'],
#                 uptime=row['uptime'],
#                 requester=row['requester'],
#                 title=row['title'],
#                 genre=row['genre'],
#                 request_type=row['request_type'],
#                 looper_slot=row['looper_slot'],
#                 looper_file_number=row['looper_file_number']
#             )

#     # Example usage
#     xlsx_file = 'onglog_2024-09-28.xlsx'
#     read_xlsx_and_insert_to_db(xlsx_file)

# import pytz

# eastern = pytz.timezone('US/Eastern')
# sydney = pytz.timezone('Australia/Sydney')

# def convert_to_sydney_time(eastern_time):
#     eastern_time = eastern.localize(eastern_time)
#     sydney_time = eastern_time.astimezone(sydney)
#     return sydney_time
