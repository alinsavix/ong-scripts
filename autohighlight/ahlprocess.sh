#!/bin/bash
set -o errtrace

error_count=0
log_error() {
    echo "WARNING: command failed on line $LINENO: $BASH_COMMAND" >&2
    ((error_count++))
}
trap 'log_error' ERR

log_exit() {
    if [[ $error_count -gt 0 ]]; then
        echo "WARNING: ${error_count} tasks failed, this (probably) shouldn't happen" >&2
        exit 1
    else
        echo "INFO: All autohighlight tasks ran successfully" >&2
    fi
}


# Actual script starts
if [[ -f "/etc/default/ong" ]]; then
    # shellcheck source=/dev/null
    source /etc/default/ong
else
    ONG_SCRIPTDIR=$(realpath "$(dirname "$0")/..")
    OBS_LOGDIR=/tmp/logs
    FFMPEG_BIN=ffmpeg
fi

AHDIR="$ONG_SCRIPTDIR/autohighlight"
cd "$AHDIR" >/dev/null 2>&1 || { echo "ERROR: script directory '$AHDIR' doesn't exist"; exit 1; }

# shellcheck source=/dev/null
source "venv/bin/activate" 2>&1 || { echo "ERROR: couldn't activate venv in '$AHDIR'"; exit 1; }

# we're initialized, set up our exit logging
trap 'log_exit' EXIT

# Update our local database of onglog entries
echo "TASK: update onglog database"
./onglog.py --credsfile "${AHDIR}/gsheets_credentials.json"

# Find all our requested highlights
echo "TASK: parse OBS logs for highlight requests"
./ahllogparse.py "$OBS_LOGDIR"/*.txt

# remux our various file types
echo "TASK: remux recorded audio & video"

# FIXME: better handling of TESSDATA_PREFIX
export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/tessdata"

# FIXME: better "check to see if these files exist and remux them if so" code
ls /ong/recorded/clean\ * >/dev/null 2>&1 && ./metaremux.py --remux-dest-dir /ong/clean --trash-dir /ong/trash /ong/recorded/clean\ *
ls /ong/recorded/stems\ * >/dev/null 2>&1 &&./metaremux.py --remux-dest-dir /ong/stems --trash-dir /ong/trash /ong/recorded/stems\ *
ls /ong/recorded/looper\ * >/dev/null 2>&1 && ./metaremux.py --remux-dest-dir /ong/looper --trash-dir /ong/trash /ong/recorded/looper\ *
ls /ong/recorded/bruce\ * >/dev/null 2>&1 && ./metaremux.py --remux-dest-dir /ong/bruce --trash-dir /ong/trash /ong/recorded/bruce\ *


# extract the bits of audio/video we care about
echo "TASK: extract highlights"
./ahlextract.py --ffmpeg-bin "$FFMPEG_BIN" --dest-dir /ong/autohighlights --source-dir /ong/clean --content-class clean --extra-head 3 --extract-length 120
./ahlextract.py --ffmpeg-bin "$FFMPEG_BIN" --dest-dir /ong/autohighlights --source-dir /ong/stems --content-class stems --extra-head 3 --extract-length 120


# generate autocrops
echo "TASK: generate autocrops"

# FIXME: come up with a better way to identify what actually needs cropping
./autocrop.py --ffmpeg-bin /usr/local/bin/ong-ffmpeg /ong/autohighlights/highlight_*_clean_????-??-??.mp4

if [[ $error_count -gt 0 ]]; then
    echo "WARNING: ${error_count} tasks failed, this (probably) shouldn't happen"
    exit 1
fi


# finally, try to restart the sync process again so that it syncs the cropped
# highlights, too, as soon as they're done processing.
systemctl try-restart ong-datasync.service || /bin/true
