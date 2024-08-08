#!/usr/bin/bash
set -e -E
DESTUSER=ongsync
# DESTHOST=tenforward
# DESTDIR=/Twitch/JonathanOng/Loopstation
SSHKEY="/root/.ssh/ongbot.key"
# SRCDIR=/ong/looper
# THROTTLE=2500   # in kB, so 2500kB = 20000kb

export PATH=/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin

SRCDIR=$1
DESTHOST=$2
DESTDIR=$3
THROTTLE=$4

if [[ -z $SRCDIR || -z $DESTHOST || -z $DESTDIR ]]; then
    echo "Usage: $0 <srcdir> <desthost> <destdir> [throttle]"
    exit 1
fi

if [[ ! -d $SRCDIR ]]; then
    echo "Source directory does not exist: $SRCDIR"
    exit 1
fi

if [[ -z $THROTTLE || $THROTTLE -lt 0 ]]; then
    THROTTLE=2500
fi

declare -a rsync_opts
rsync_opts=(
    # Next line is same as --archive, minus --perms
    '--recursive' '--links' '--times' '--group' '--owner'
    '--chmod=0644'
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

nocache rsync "${rsync_opts[@]}" "${SRCDIR}/" "${DESTUSER}@${DESTHOST}:${DESTDIR}/"
