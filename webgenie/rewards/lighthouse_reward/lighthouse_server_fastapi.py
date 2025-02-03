import bittensor as bt
import os
import sys
import threading
import uvicorn
import psutil
import signal
import subprocess

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webgenie.constants import (
    WORK_DIR,
    LIGHTHOUSE_SERVER_WORK_DIR,
    LIGHTHOUSE_SERVER_PORT,
)
from webgenie.helpers.ports import kill_process_on_port
kill_process_on_port(LIGHTHOUSE_SERVER_PORT)

app = FastAPI()
static_folder = f"/{LIGHTHOUSE_SERVER_WORK_DIR}"
lighthouse_server_thread = None

def make_work_dir():
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
        bt.logging.info(f"Created work directory at {WORK_DIR}")

    if not os.path.exists(LIGHTHOUSE_SERVER_WORK_DIR):
        os.makedirs(LIGHTHOUSE_SERVER_WORK_DIR)
        bt.logging.info(f"Created lighthouse server work directory at {LIGHTHOUSE_SERVER_WORK_DIR}")

make_work_dir()
app.mount("/", StaticFiles(directory=f"{LIGHTHOUSE_SERVER_WORK_DIR}"), name="static")


def stop_lighthouse_server():
    global lighthouse_server_thread
    if lighthouse_server_thread:
        lighthouse_server_thread.join(10)
        lighthouse_server_thread = None
        bt.logging.info("Lighthouse server stopped")


def start_lighthouse_server():
    try:
        bt.logging.success(f"Trying to start lighthouse server on port {LIGHTHOUSE_SERVER_PORT}")
        uvicorn.run(app, host="0.0.0.0", port=LIGHTHOUSE_SERVER_PORT, log_level="error")
    except Exception as e:
        bt.logging.error(f"Error starting lighthouse server: {e}")
        stop_lighthouse_server()
        sys.exit(1)



def start_lighthouse_server_thread():
    global lighthouse_server_thread
    try:
        lighthouse_server_thread = threading.Thread(target=start_lighthouse_server, daemon=True)
        lighthouse_server_thread.start()
        bt.logging.info("Lighthouse server started")
    except Exception as e:
        bt.logging.error(f"Error starting lighthouse server: {e}")
        stop_lighthouse_server()
        sys.exit(1)
