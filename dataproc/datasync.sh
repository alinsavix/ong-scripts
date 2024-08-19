#!/usr/bin/bash
DESTUSER=ongsync
DESTHOST=tenforward
# DESTDIR=/Twitch/JonathanOng/Loopstation
SSHKEY="/root/.ssh/ongbot.key"
# SRCDIR=/ong/looper
THROTTLE=3000   # in kB, so 2500kB = 20000kb
THROTTLE=0

export PATH=/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin

SRCDIR="$1"
DESTDIR="$2"

if [ -z "$SRCDIR" ] || [ -z "$DESTDIR" ]; then
    echo "Usage: $0 SRCDIR DESTDIR" >&2
    exit 1
fi

if [ ! -d "$SRCDIR" ]; then
    echo "'$SRCDIR' does not exist, or is not a directory" >&2
    exit 1
fi

declare -a rsync_opts
rsync_opts=(
    # Next line is same as --archive, minus --perms
    '--recursive' '--links' '--times' '--group' '--owner'
    '--chmod=0664'
    '--omit-dir-times'  # otherwise complains about perms for '.'
    '--partial'    # keep partially transferred files (so we can restart)
    '--append'     # add data if files have grown, rather than rewriting
    '--verbose' '--progress'
)

if [[ -n $THROTTLE && $THROTTLE -gt 0 ]]; then
    rsync_opts+=(
        "--bwlimit=${throttle}"   # bandwidth limit (kB/sec)
    )
fi

declare -a rsh_opts
rsh_opts=(
    'ssh'

    # So that host key changes don't break copies (not ideal, but convenient)
    '-o' 'BatchMode=yes'
    '-o' 'StrictHostKeyChecking=no'
    '-o' 'UserKnownHostsFile=/dev/null'

    # make network fails not hang forever
    '-o' 'ServerAliveInterval=5'
    '-o' 'ServerAliveCountMax=3'

    # Which private key to use
    '-i' "${SSHKEY}"

    # quiet!
    '-q'
)

rsync_opts+=(
    "--rsh=${rsh_opts[*]}"
)

# rsync_opts+=(
#    "--include=/${type}/***"
#    '--exclude=*'
#)
rsync_opts+=(
    '--exclude=tmp*'
#    '--exclude=.??*'
)

echo "syncing ${SRCDIR} to ${DESTHOST}:${DESTDIR}"
nocache rsync "${rsync_opts[@]}" "${SRCDIR}/" "${DESTUSER}@${DESTHOST}:${DESTDIR}/"
