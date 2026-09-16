#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import google.auth.transport.requests
import googleapiclient.discovery
import googleapiclient.errors
import googleapiclient.http
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from tdvutil.argparse import CheckFile

# NOTE: Unlike vodprep, this can't use a service account -- youtube won't let
# one act on a channel, because there's no way to give a service account
# ownership of one. You need an OAuth "Desktop app" client from the same Google
# cloud project (with the YouTube Data API v3 enabled), saved as
# client_secret.json. The first run will pop up a browser to get your blessing,
# and cache the resulting token in youtube_token.json.


# We need to edit video metadata, set thumbnails, and add to playlists
# which these scopes cover
SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

# Videos get uploaded by hand, and youtube names them after the file they came
# from -- but it mangles the name on the way in, turning dashes, dots and
# colons into spaces, so "JonathanOng 2026-08-21 v2852039660-seg1.mp4" arrives
# as "JonathanOng 2026 08 21 v2852039660 ... seg1". Hence being relaxed about
# separators. Trimmed videos have a timecode range in the middle of all that,
# which we don't care about.
UPLOAD_TITLE_RE = re.compile(
    r"^JonathanOng\s+(?P<date>\d{4}[-\s]\d{2}[-\s]\d{2})\s+v(?P<vodid>\d+)(?P<rest>.*)$")
SEGMENT_RE = re.compile(r"\bseg(?P<seg>\d+)\b")

# Any trailing "(something)" on a generated title, which we need to shuffle
# around when a stream is split over multiple videos
ANNOTATION_RE = re.compile(r"\s+\([^()]*\)$")

# The playlist a given year's VODs belong in
PLAYLIST_TITLE = "Jonathan Ong - {year} VODs"

# We only started doing this part by script recently, and everything before
# was published by hand, so make sure when we're looking for what needs to
# be published still, don't go back too far.
EARLIEST_DATE = "2026-07-28"

# For printing out what we did at the end of a run
VIDEO_URL = "https://youtu.be/{video_id}"
VIDEO_PRIVACY = "unlisted"

# The line in the info file that starts the bit we use as the timestamped
# track list
TRACKLIST_MARKER = "APPROXIMATE start times of each segment:"

# Boilerplate to stick on the front of every description, if it exists. It can
# use {date} and {title} to refer to the stream it's going on.
DESC_HEADER_FILE = Path("description_header.txt")

# Where to track what video goes with what stream.
STATE_FILE = Path("youtube_videos.json")


# Pull the stream date and the part number out of an upload's title, or None
# if it doesn't look like one of ours at all.
def parse_upload_title(title: str) -> Optional[Tuple[str, int]]:
    m = UPLOAD_TITLE_RE.match(title)
    if not m:
        return None

    seg = SEGMENT_RE.search(m.group("rest"))
    return m.group("date").replace(" ", "-"), int(seg.group("seg")) if seg else 1


# Everything we generated for a single stream, ready to be pushed at youtube
class StreamInfo(NamedTuple):
    date: str
    title: str
    description: str
    thumbnail: Path


# One of the videos that make up a stream. thumb_sha is the thumbnail we last
# put on it, so that a re-run doesn't upload the same image over and over.
class Video(NamedTuple):
    video_id: str
    part: int
    orig_title: str
    thumb_sha: str = ""


def file_sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


# Pull the title and the timestamp list back out of the txt file made
# by vodprep. The file is meant for humans (it's what we used to paste in by
# hand), so there's a "TITLE:" line and a trailing note about the thumbnail
# that we don't want in the description.
def read_stream_info(datestr: str) -> StreamInfo:
    infofile = Path(f"{datestr}.txt")
    thumbnail = Path(f"{datestr}.jpg")

    if not infofile.exists():
        raise FileNotFoundError(f"no info file {infofile}, has vodprep been run?")
    if not thumbnail.exists():
        raise FileNotFoundError(f"no thumbnail {thumbnail}, has vodprep been run?")

    lines = infofile.read_text(encoding="utf-8").splitlines()

    titles = [x[len("TITLE: "):].strip() for x in lines if x.startswith("TITLE: ")]
    if not titles:
        raise ValueError(f"{infofile} has no TITLE: line in it")

    try:
        first = lines.index(TRACKLIST_MARKER)
    except ValueError:
        raise ValueError(f"{infofile} has no timestamp list in it")

    last = next(i for i, x in enumerate(lines) if x.startswith("TITLE: "))
    desc = "\n".join(lines[first:last]).strip()

    return StreamInfo(date=datestr, title=titles[0], description=desc, thumbnail=thumbnail)


# Make the description for one video: the boilerplate header (if we
# have one), and then the list of songs, which only belongs on the video that
# has the start of the stream in it. Returns None if we end up with nothing
# to say, in which case we leave the description alone.
def build_description(info: StreamInfo, header: str, with_tracklist: bool) -> Optional[str]:
    chunks = []

    if header:
        # deliberately not str.format(), so that a stray brace in the
        # boilerplate doesn't blow up in our face
        chunks.append(header.replace("{date}", info.date)
                            .replace("{title}", info.title).strip())

    if with_tracklist:
        chunks.append(info.description)

    if not chunks:
        return None

    return "\n\n".join(chunks)


# Work out the title for one part of a stream. A stream that fit in a single
# video just gets the title as generated. When it didn't, every part gets a
# ", Part N", and we have to think about the annotation on the end.
def part_title(title: str, part: int, nparts: int) -> str:
    if nparts <= 1:
        return title

    annotation = ""
    m = ANNOTATION_RE.search(title)
    if m:
        annotation = m.group(0)
        title = title[:m.start()]

        # "(incl. Concert Grand)" means the concert grand happened at the end
        # of the stream, so it belongs to whichever video the end of the stream
        # landed in. Anything else ("Timer'd Concert Grand Stream", say)
        # describes the whole stream, so every part keeps it.
        if "incl." in annotation and part != nparts:
            annotation = ""

    return f"{title}, Part {part}{annotation}"


def load_state(statefile: Path) -> Dict[str, List[Video]]:
    if not statefile.exists():
        return {}

    raw = json.loads(statefile.read_text(encoding="utf-8"))
    return {date: [Video(**v) for v in vids] for date, vids in raw.items()}


def save_state(statefile: Path, state: Dict[str, List[Video]]) -> None:
    raw = {date: [v._asdict() for v in vids] for date, vids in state.items()}
    statefile.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_youtube(client_secret: Path, tokenfile: Path) -> Any:
    creds = None
    if tokenfile.exists():
        creds = Credentials.from_authorized_user_file(str(tokenfile), SCOPES)

    if creds and not creds.valid and creds.expired and creds.refresh_token:
        try:
            creds.refresh(google.auth.transport.requests.Request())
        except RefreshError:
            print("WARNING: cached youtube token wouldn't refresh, asking again",
                  file=sys.stderr)
            creds = None

    if not creds or not creds.valid:
        if not client_secret.exists():
            raise FileNotFoundError(
                f"{client_secret} not found -- you need an OAuth 'Desktop app' client"
                " from a Google cloud project with the YouTube Data API v3 enabled")

        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
        creds = flow.run_local_server(port=0)

    tokenfile.write_text(creds.to_json(), encoding="utf-8")
    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)


# Find the uploads that still look like freshly uploaded VOD files, keyed by
# the stream date in their title.
def find_uploads(yt: Any, max_uploads: int) -> Dict[str, List[Video]]:
    chan = yt.channels().list(part="contentDetails", mine=True).execute()
    if not chan.get("items"):
        raise RuntimeError("no channel associated with these credentials")

    uploads_id = chan["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    found: Dict[str, List[Video]] = {}
    seen = 0
    page = None

    while seen < max_uploads:
        resp = yt.playlistItems().list(
            part="snippet", playlistId=uploads_id, maxResults=50, pageToken=page).execute()

        for item in resp.get("items", []):
            seen += 1
            title = item["snippet"]["title"]

            parsed = parse_upload_title(title)
            if parsed is None:
                continue

            date, part = parsed
            found.setdefault(date, []).append(Video(
                video_id=item["snippet"]["resourceId"]["videoId"],
                part=part,
                orig_title=title,
            ))

        page = resp.get("nextPageToken")
        if not page:
            break

    return found


# For when a video we expected to match didn't. Dumps what the API will
# actually tell us about, by both of the ways we can ask.
def list_uploads(yt: Any, max_uploads: int) -> None:
    chan = yt.channels().list(part="contentDetails,snippet", mine=True).execute()
    if not chan.get("items"):
        raise RuntimeError("no channel associated with these credentials")

    channel = chan["items"][0]
    uploads_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]
    print(f"channel: {channel['snippet']['title']!r} ({channel['id']})")
    print(f"uploads playlist: {uploads_id}")
    print()

    seen = 0
    matched = 0
    page = None
    in_uploads = set()

    while seen < max_uploads:
        resp = yt.playlistItems().list(
            part="snippet,status", playlistId=uploads_id, maxResults=50,
            pageToken=page).execute()

        for item in resp.get("items", []):
            seen += 1
            title = item["snippet"]["title"]
            video_id = item["snippet"]["resourceId"]["videoId"]
            privacy = item.get("status", {}).get("privacyStatus", "?")
            in_uploads.add(video_id)

            parsed = parse_upload_title(title)
            if parsed is None:
                extra = ""
            else:
                matched += 1
                extra = f" -> date={parsed[0]} part={parsed[1]}"

            print(f"  [{'MATCH' if parsed else 'no':^5}] {video_id}  {privacy:<9}"
                  f" {title!r}{extra}")

        page = resp.get("nextPageToken")
        if not page:
            break

    print()
    print(f"{seen} upload(s) visible, {matched} of them look like unpublished VODs")

    # The uploads playlist doesn't necessarily show everything, so ask the
    # other way too and see if it turns up anything extra.
    print()
    print("cross-checking against search.list(forMine=True)...")
    resp = yt.search().list(
        part="snippet", forMine=True, type="video", order="date", maxResults=50).execute()

    extras = 0
    for item in resp.get("items", []):
        video_id = item["id"]["videoId"]
        if video_id in in_uploads:
            continue

        extras += 1
        print(f"  [only here] {video_id}  {item['snippet']['title']!r}")

    if extras:
        print(f"{extras} video(s) search can see that the uploads playlist can't")
    else:
        print("nothing there that the uploads playlist didn't already have")


def find_playlist(yt: Any, year: int, cache: Dict[int, Optional[str]]) -> Optional[str]:
    if year in cache:
        return cache[year]

    want = PLAYLIST_TITLE.format(year=year)
    cache[year] = None
    page = None

    while True:
        resp = yt.playlists().list(
            part="snippet", mine=True, maxResults=50, pageToken=page).execute()

        for playlist in resp.get("items", []):
            if playlist["snippet"]["title"] == want:
                cache[year] = playlist["id"]
                return playlist["id"]

        page = resp.get("nextPageToken")
        if not page:
            return None


def in_playlist(yt: Any, playlist_id: str, video_id: str) -> bool:
    resp = yt.playlistItems().list(
        part="id", playlistId=playlist_id, videoId=video_id, maxResults=1).execute()
    return len(resp.get("items", [])) > 0


def get_video(yt: Any, video_id: str) -> Dict[str, Any]:
    resp = yt.videos().list(part="snippet,status", id=video_id).execute()
    if not resp.get("items"):
        raise RuntimeError(f"video {video_id} doesn't exist (or isn't ours)")

    return resp["items"][0]


# Both "snippet" and "status" get *replaced* by an update rather than merged
# into, so anything we don't hand back gets quietly wiped off the video. Hence
# all the copying of things we don't actually care about.
def update_video(yt: Any, video_id: str, old: Dict[str, Any], title: str,
                 description: Optional[str], privacy: Optional[str]) -> None:
    old_snippet = old["snippet"]
    old_status = old.get("status", {})

    snippet: Dict[str, Any] = {
        "title": title,
        "description": description if description is not None
                       else old_snippet.get("description", ""),
        "categoryId": old_snippet["categoryId"],
    }

    for key in ("tags", "defaultLanguage", "defaultAudioLanguage"):
        if key in old_snippet:
            snippet[key] = old_snippet[key]

    body: Dict[str, Any] = {"id": video_id, "snippet": snippet}
    parts = "snippet"

    # Only touch the privacy if it's actually wrong, so that we're not putting
    # the made-for-kids flag at risk on videos that are already fine.
    if privacy is not None and old_status.get("privacyStatus") != privacy:
        status: Dict[str, Any] = {"privacyStatus": privacy}

        for key in ("license", "embeddable", "publicStatsViewable",
                    "selfDeclaredMadeForKids"):
            if key in old_status:
                status[key] = old_status[key]

        # youtube only hands "selfDeclaredMadeForKids" back to the video's
        # owner, so fall back to the derived flag if we didn't get it
        if "selfDeclaredMadeForKids" not in status and "madeForKids" in old_status:
            status["selfDeclaredMadeForKids"] = old_status["madeForKids"]

        body["status"] = status
        parts = "snippet,status"

    yt.videos().update(part=parts, body=body).execute()


def set_thumbnail(yt: Any, video_id: str, thumbnail: Path) -> None:
    yt.thumbnails().set(
        videoId=video_id,
        media_body=googleapiclient.http.MediaFileUpload(
            str(thumbnail), mimetype="image/jpeg")).execute()


# Do the actual work for one stream, or at least say what we would do.
#
# FIXME: make the logging less verbose/more selectable
def publish_stream(yt: Any, args: argparse.Namespace, info: StreamInfo,
                   videos: List[Video], playlists: Dict[int, Optional[str]],
                   header: str) -> List[Video]:
    nparts = len(videos)
    year = int(info.date[:4])
    want_thumb = file_sha(info.thumbnail)
    done = []

    playlist_id = find_playlist(yt, year, playlists)
    if playlist_id is None:
        print(f"  WARNING: no {PLAYLIST_TITLE.format(year=year)!r} playlist found,"
              " not adding to one", file=sys.stderr)

    privacy = None if args.privacy == "keep" else args.privacy

    for video in videos:
        title = part_title(info.title, video.part, nparts)
        done.append(video)

        # Only the video with the start of the stream in it wants the list of
        # songs, since the timestamps would be wrong for any later part. The
        # boilerplate goes on all of them.
        description = build_description(info, header, with_tracklist=video.part == 1)

        old = get_video(yt, video.video_id)
        old_snippet = old["snippet"]
        old_status = old.get("status", {})

        print(f"  part {video.part} of {nparts}: {video.video_id} ({video.orig_title})")

        if old_snippet["title"] == title:
            print(f"    title: already {title!r}")
        else:
            print(f"    title: {old_snippet['title']!r} -> {title!r}")

        if description is None:
            print("    description: left alone (not the first part, no boilerplate)")
        elif old_snippet.get("description", "") == description:
            print("    description: already up to date")
        else:
            if old_snippet.get("description"):
                print(f"    description: replacing {len(old_snippet['description'])}"
                      " characters of existing description:")
                for line in old_snippet["description"].splitlines():
                    print(f"      | {line}")
            what = "boilerplate + timestamps" if video.part == 1 else "boilerplate"
            print(f"    description: {len(description)} characters of {what}")

        thumb_changed = video.thumb_sha != want_thumb
        if thumb_changed:
            print(f"    thumbnail: setting to {info.thumbnail}")
        else:
            print(f"    thumbnail: already {info.thumbnail}")

        if privacy is None:
            print("    privacy: left alone")
        elif old_status.get("privacyStatus") == privacy:
            print(f"    privacy: already {privacy}")
        else:
            print(f"    privacy: {old_status.get('privacyStatus')} -> {privacy}")

        playlist_add = False
        if playlist_id is not None:
            if in_playlist(yt, playlist_id, video.video_id):
                print("    playlist: already in it")
            else:
                print(f"    playlist: adding to {PLAYLIST_TITLE.format(year=year)!r}")
                playlist_add = True

        # Skip any call that wouldn't actually change anything -- an update
        # and a thumbnail are 50 quota units each, which adds up fast when
        # you're re-running over a backlog.
        snippet_changed = any([
            old_snippet["title"] != title,
            description is not None and old_snippet.get("description", "") != description,
            privacy is not None and old_status.get("privacyStatus") != privacy,
        ])

        if not (snippet_changed or thumb_changed or playlist_add):
            print("    nothing to do")
            continue

        if not args.go:
            continue

        if snippet_changed:
            update_video(yt, video.video_id, old, title, description, privacy)

        if thumb_changed:
            set_thumbnail(yt, video.video_id, info.thumbnail)
            done[-1] = video._replace(thumb_sha=want_thumb)

        if playlist_add:
            yt.playlistItems().insert(part="snippet", body={"snippet": {
                "playlistId": playlist_id,
                "resourceId": {"kind": "youtube#video", "videoId": video.video_id},
            }}).execute()

        print("    updated")

    return done


# Which streams are we being asked about? Either the dates on the command line,
# or everything we have generated info for but never published. Either way we
# put the oldest ones first; playlist inserts go on the end of the playlist,
# so processing in this order is what puts the playlist in order.
def dates_to_do(args: argparse.Namespace, state: Dict[str, List[Video]]) -> List[str]:
    if args.dates:
        return sorted(set(args.dates))

    infofiles = Path(".").glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].txt")
    dates = sorted(p.stem for p in infofiles if p.stem not in state)

    # ISO-8601 dates sort correctly as strings, convenient!
    since = max(EARLIEST_DATE, args.since) if args.since else EARLIEST_DATE

    return [d for d in dates if d >= since]


def datestr(arg_value: str) -> str:
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", arg_value):
        raise argparse.ArgumentTypeError(f"{arg_value!r} is not a YYYY-MM-DD date")

    return arg_value


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill in the title, description, thumbnail, and playlist for"
                    " VODs we've generated info for",
        allow_abbrev=True,
    )

    parser.add_argument(
        "--client-secret",
        default="client_secret.json",
        type=Path,
        action=CheckFile(must_exist=True),
        help="OAuth client secrets file for youtube",
    )

    parser.add_argument(
        "--token-file",
        default=Path("youtube_token.json"),
        type=Path,
        help="where to cache our youtube credentials",
    )

    parser.add_argument(
        "--state-file",
        default=STATE_FILE,
        type=Path,
        help="file remembering which youtube video went with which stream",
    )

    parser.add_argument(
        "--privacy",
        default=VIDEO_PRIVACY,
        choices=["unlisted", "private", "public", "keep"],
        help="what the videos' privacy should be set to (default: %(default)s)",
    )

    parser.add_argument(
        "--description-header",
        default=DESC_HEADER_FILE,
        type=Path,
        help="file of boilerplate to put above the song list in the description"
             " (default: %(default)s, if it exists)",
    )

    parser.add_argument(
        "--since",
        default=None,
        type=datestr,
        metavar="YYYY-MM-DD",
        help=f"when no dates are given, only consider streams from this one on"
             f" (never reaches back past {EARLIEST_DATE})",
    )

    parser.add_argument(
        "--max-uploads",
        default=200,
        type=int,
        help="how far back through the channel uploads to look for a match",
    )

    parser.add_argument(
        "--list-uploads",
        default=False,
        action="store_true",
        help="just list the uploads we can see and whether they'd match, then stop",
    )

    parser.add_argument(
        "--go",
        default=False,
        action="store_true",
        help="actually change things (without this we only say what we would do)",
    )

    parser.add_argument(
        "dates",
        type=datestr,
        nargs="*",
        metavar="YYYY-MM-DD",
        help="streams to publish, oldest first whatever order you list them in."
             f" Naming a date here is the only way to process one older than"
             f" {EARLIEST_DATE} (default: everything not published yet)",
    )

    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_arguments(argv[1:])

    if args.list_uploads:
        try:
            yt = get_youtube(args.client_secret, args.token_file)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        list_uploads(yt, args.max_uploads)
        return 0

    state = load_state(args.state_file)
    dates = dates_to_do(args, state)
    if not dates:
        print("No streams waiting to be published, nothing to do")
        return 0

    # The default boilerplate file is optional, but if we were pointed at a
    # specific one, it had better be there
    header = ""
    if args.description_header.exists():
        header = args.description_header.read_text(encoding="utf-8")
        print(f"Using description boilerplate from {args.description_header}")
    elif args.description_header != DESC_HEADER_FILE:
        print(f"ERROR: no such description header file: {args.description_header}",
              file=sys.stderr)
        return 1

    print(f"{len(dates)} stream(s) to look at: {dates[0]} .. {dates[-1]}")

    if not args.go:
        print("DRY RUN -- nothing will actually change. Use --go to do it for real.\n")

    try:
        yt = get_youtube(args.client_secret, args.token_file)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Only worth listing out the channel's uploads if there's a stream we
    # haven't already matched up to a video
    uploads: Optional[Dict[str, List[Video]]] = None
    playlists: Dict[int, Optional[str]] = {}

    failed = 0
    unmatched = 0
    processed: List[Tuple[str, str]] = []
    for date in dates:
        print(f"{date}:")

        try:
            info = read_stream_info(date)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failed += 1
            continue

        if date in state:
            videos = sorted(state[date], key=lambda v: v.part)
            print(f"  matched to {len(videos)} video(s) on a previous run")
        else:
            if uploads is None:
                uploads = find_uploads(yt, args.max_uploads)

            videos = sorted(uploads.get(date, []), key=lambda v: v.part)
            if not videos:
                print("  no matching upload found, skipping")
                unmatched += 1
                continue

            if len(videos) > 1 and not all(SEGMENT_RE.search(v.orig_title) for v in videos):
                print(f"  ERROR: {len(videos)} videos for this date, but they aren't"
                      " all marked with -segN, so we can't tell what order they go in",
                      file=sys.stderr)
                failed += 1
                continue

            # Number the parts 1..n, whatever the segment numbering in the
            # filenames happened to be
            videos = [v._replace(part=i + 1) for i, v in enumerate(videos)]

            # Write this down *before* we touch anything: retitling is exactly
            # what stops us being able to find these videos again
            if args.go:
                state[date] = videos
                save_state(args.state_file, state)

        try:
            videos = publish_stream(yt, args, info, videos, playlists, header)
        except (googleapiclient.errors.HttpError, RuntimeError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failed += 1
            continue

        if args.go:
            state[date] = videos
            save_state(args.state_file, state)

        for video in videos:
            processed.append((part_title(info.title, video.part, len(videos)),
                              VIDEO_URL.format(video_id=video.video_id)))

    if unmatched:
        print()
        print(f"{unmatched} stream(s) had no matching upload. Run with"
              " --list-uploads to see what is actually visible on the channel.")

    if processed:
        print()
        print(f"{len(processed)} video(s) {'processed' if args.go else 'to process'}:")
        for title, url in processed:
            print(f"  {url}  {title}")

        # Jon pastes these into Discord, where a bare URL expands into a big
        # preview embed. Wrapping each in <> tells Discord to leave it alone
        # for when Alinsa sends them to Jon. Only worth it for a real run --
        # in a dry run these URLs don't point at anything published yet.
        if args.go:
            print()
            print("For Jon:")
            for _, url in processed:
                print(f"<{url}>")

    if failed:
        print(f"{failed} stream(s) had problems", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    # make sure our output streams are properly encoded so that we can
    # not screw up Frédéric Chopin's name and such, and keep them line
    # buffered so progress shows up as it happens.
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8", line_buffering=True)

    sys.exit(main(sys.argv))
