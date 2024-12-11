#!/bin/bash
BASEDIR=/ong

echo "Performing cleanup of old stream recordings (and other artifacts)"
echo "Disk usage before cleanup:"
df -h "$BASEDIR"

# the trash directory is stuff that has been remuxed, so we don't need to
# keep it as long
find "${BASEDIR}/trash" -type f -mtime +7 -delete

# Everything else, keep for a bit longer, so we have a chance to notice
# if they're not syncing or something.
for d in autohighlights bruce clean looper stems
do
    find "${BASEDIR}/${d}" -type f -mtime +21 -delete
done

echo "Cleanup complete"
echo "Disk usage after cleanup:"
df -h "$BASEDIR"
