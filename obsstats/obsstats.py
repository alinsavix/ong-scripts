#!/usr/bin/env -S uv run --script
# Do some OBS monitoring things running under inputs.execd in telegraf
import argparse
import asyncio
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import simpleobsws
from tdvutil import hms_to_sec, ppretty


@dataclass
class OBSState:
    """Shared state between OBS monitor task and stdin handler"""
    connected: bool = False
    version: str = "unknown"
    stats: Dict = field(default_factory=dict)
    outputs: Dict[str, Dict] = field(default_factory=dict)
    last_update: int = 0


DEBUG = False

def now():
    return int(time.time())

def log(msg: str) -> None:
    """Always log important messages"""
    print(msg, file=sys.stderr)
    sys.stderr.flush()

def debug_log(msg: str) -> None:
    """Log only when debug mode is enabled"""
    if DEBUG:
        print(msg, file=sys.stderr)
        sys.stderr.flush()

def printmetric(metric_name: str, ts: int, value: int | float, tags: Dict[str, str] | None = None):
    tags_list = ""
    if tags:
        tags_list = " ".join([f"{k}={v}" for k, v in tags.items()])

    # we have to do this the roundabout way because if we use \n normally,
    # it will output newline + carriage return on windows, and the wavefront
    # input plugin on telegraf will explode on the \r character (sigh)
    sys.stdout.buffer.write(
        f"obs.{metric_name} {value} {ts} {tags_list}\n".encode())

def normalize_name(name: str) -> str:
    name = re.sub(r"[^\w-]+", "_", name)
    return name.lower()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gather OBS stats with a telegraf input processor")

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="address or hostname of host running OBS"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=4455,
        help="port number for OBS websocket"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug logging to stderr"
    )

    parsed_args = parser.parse_args()

    # Set global debug flag
    global DEBUG
    DEBUG = parsed_args.debug

    return parsed_args


async def main():
    # make library logging be quiet
    logging.basicConfig(level=logging.FATAL)

    args = parse_args()

    while True:
        try:
            await run(args)
            # If run() returns normally (stdin closed), exit cleanly
            log("stdin closed, exiting program")
            break
        except KeyboardInterrupt:
            log("Keyboard interrupt, exiting")

            printmetric("active", now(), 0, {})
            sys.stdout.flush()
            break
        except ConnectionError as e:
            log(f"OBS connection error, trying again in 60 seconds: {e}")
            await asyncio.sleep(60)
        except Exception as e:
            log(f"UNKNOWN EXCEPTION: {ppretty(e)}")
            await asyncio.sleep(60)


async def obs_monitor_task(args: argparse.Namespace, state: OBSState, shutdown_event: asyncio.Event):
    """Background task that monitors OBS and updates shared state"""
    while not shutdown_event.is_set():
        ws = simpleobsws.WebSocketClient(
            url=f'ws://{args.host}:{args.port}',
            password=''
        )

        try:
            await ws.connect()
            await ws.wait_until_identified()
            state.connected = True
            debug_log(f"Connected to OBS at {args.host}:{args.port}")
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
            state.connected = False
            state.stats = {}
            state.outputs = {}

            await asyncio.sleep(5)
            continue

        async def on_exit_started(event_type, event_data):
            log("Got OBS exit signal, disconnecting")
            state.connected = False
            shutdown_event.set()

        ws.register_event_callback(on_exit_started, 'ExitStarted')

        try:
            request = simpleobsws.Request('GetVersion')
            ret = await ws.call(request, timeout=5)
            if ret.ok():
                state.version = ret.responseData.get("obsVersion", "unknown")
        except Exception as e:
            log(f"Failed to get version: {e}")
            state.version = "unknown"

        # get us some stats regularly
        while state.connected and not shutdown_event.is_set():
            try:
                # main stats
                request = simpleobsws.Request('GetStats')
                ret = await ws.call(request, timeout=5)
                if ret.ok():
                    state.stats = ret.responseData
                    state.last_update = now()

                # per-output stats - We're using a fixed list of outputs here,
                # because OBS was periodically crashing if we iterated the
                # output list on a regular basis (even if it wasn't changing)
                outputs = ["simple_stream", "simple_file_output",
                           "adv_stream", "adv_file_output"]
                for output_name in outputs:
                    try:
                        request = simpleobsws.Request(
                            'GetOutputStatus', {'outputName': output_name})
                        ret = await ws.call(request, timeout=5)
                        if ret.ok():
                            state.outputs[output_name] = ret.responseData
                    except Exception:
                        # Remove from cache if we can't get status
                        state.outputs.pop(output_name, None)

                # FIXME: make configurable?
                await asyncio.sleep(15)

            except Exception as e:
                log(f"Error updating OBS stats: {e}")
                state.connected = False
                break

        # If we got here, our connection to OBS is in an unhappy state
        try:
            await ws.disconnect()
        except:
            pass

        if not shutdown_event.is_set():
            await asyncio.sleep(5)


async def stdin_handler_task(args: argparse.Namespace, state: OBSState, shutdown_event: asyncio.Event):
    debug_log("stdin_handler_task started")

    stdin_queue = asyncio.Queue()  # queue for stdin lines

    loop = asyncio.get_event_loop()

    # AFAIK there's no decent way to do a non-blocking read of stdin in
    # python, and definitely not in a cross-platform way, so we'll make
    # a separate thread for doing those reads.
    def stdin_reader_thread():
        try:
            while not shutdown_event.is_set():
                line = sys.stdin.readline()
                if len(line) == 0:
                    asyncio.run_coroutine_threadsafe(stdin_queue.put(None), loop)
                    break
                # Put the line in the queue (not that we actually care about
                # the content, just the fact that we got a line)
                asyncio.run_coroutine_threadsafe(stdin_queue.put(line), loop)
        except Exception as e:
            log(f"stdin_reader_thread error: {e}")
            asyncio.run_coroutine_threadsafe(stdin_queue.put(None), loop)

    import threading
    reader_thread = threading.Thread(target=stdin_reader_thread, daemon=True)
    reader_thread.start()

    count = 0

    while not shutdown_event.is_set():
        try:
            x = await asyncio.wait_for(stdin_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # Just keep looping just keep looping
            continue
        except Exception as e:
            log(f"Error reading from stdin queue: {e}")
            break

        if x is None:
            log("stdin closed, exiting")
            shutdown_event.set()
            break

        count += 1
        debug_log(
            f"Received request #{count}, connected={state.connected}, last_update={state.last_update}")

        ts = state.last_update if state.last_update > 0 else now()

        # Don't output stale data (older than 45 seconds)
        data_age = now() - state.last_update if state.last_update > 0 else 9999999
        data_is_stale = data_age >= 45

        if not state.connected or data_is_stale:
            # no metrics to give, just send active=0 with current timestamp
            debug_log(
                f"Emitting active=0 (connected={state.connected}, data_is_stale={data_is_stale})")
            printmetric("active", now(), 0, {})
            sys.stdout.flush()
            continue

        # else we're connected and have current stats
        debug_log(f"Emitting metrics (age={data_age:.1f}s)")
        printmetric("active", ts, 1, {})

        tags = {"version": state.version}

        r = state.stats
        if r:
            printmetric("usage.cpu_pct", ts, r.get("cpuUsage", 0), tags)
            printmetric("usage.memory_mb", ts, r.get("memoryUsage", 0), tags)
            printmetric("fps", ts, r.get("activeFps", 0), tags)
            printmetric("frames.render.time_avg_ms", ts,
                        r.get("averageFrameRenderTime", 0), tags)
            printmetric("frames.render.skipped", ts, r.get("renderSkippedFrames", 0), tags)
            printmetric("frames.render.total", ts, r.get("renderTotalFrames", 0), tags)
            printmetric("frames.output.skipped", ts, r.get("outputSkippedFrames", 0), tags)
            printmetric("frames.output.total", ts, r.get("outputTotalFrames", 0), tags)
            printmetric("websocket.messages.incoming", ts,
                        r.get("webSocketSessionIncomingMessages", 0), tags)
            printmetric("websocket.messages.outgoing", ts,
                        r.get("webSocketSessionOutgoingMessages", 0), tags)

        # per-output stats
        for output_name, output_data in state.outputs.items():
            output_tags = {"output": normalize_name(output_name)}
            printmetric("output.active", ts, 1 if output_data.get(
                "outputActive", False) else 0, tags | output_tags)
            printmetric("output.reconnecting", ts,
                        1 if output_data.get("outputReconnecting", False) else 0, tags | output_tags)

            # no reason to dump stats often for something that's not active or reconnecting
            if not any([output_data.get("outputActive", False), output_data.get("outputReconnecting", False)]) and count % 10 > 0:
                continue

            printmetric("output.duration_s", ts, output_data.get(
                "outputDuration", 0) / 1000, tags | output_tags)
            printmetric("output.congestion", ts, output_data.get(
                "outputCongestion", 0), tags | output_tags)
            printmetric("output.bytes", ts, output_data.get(
                "outputBytes", 0), tags | output_tags)
            printmetric("output.frames.skipped", ts,
                        output_data.get("outputSkippedFrames", 0), tags | output_tags)
            printmetric("output.frames.total", ts, output_data.get(
                "outputTotalFrames", 0), tags | output_tags)

        sys.stdout.flush()


async def run(args: argparse.Namespace):
    shutdown_event = asyncio.Event()
    state = OBSState()

    # One task for wrangling OBS, one for wrangling stdin
    monitor_task = asyncio.create_task(obs_monitor_task(args, state, shutdown_event))
    stdin_task = asyncio.create_task(stdin_handler_task(args, state, shutdown_event))

    await asyncio.gather(monitor_task, stdin_task)

    printmetric("active", now(), 0, {})
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Clean exit on Ctrl-C, no traceback
        pass
