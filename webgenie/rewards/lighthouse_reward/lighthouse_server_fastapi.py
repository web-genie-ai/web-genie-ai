import bittensor as bt
import os
import threading
import uvicorn

from fastapi import FastAPI
from fastapi.responses import FileResponse

from webgenie.constants import (
    LIGHTHOUSE_SERVER_WORK_DIR,
    LIGHTHOUSE_SERVER_PORT,
)


app = FastAPI()
static_folder = f"/{LIGHTHOUSE_SERVER_WORK_DIR}"
lighthouse_server_thread = None


@app.get("/{file_path:path}")
async def serve_file(file_path: str):
    try:
        print("serving file", file_path)
        return FileResponse(os.path.join(static_folder, file_path))
    except Exception as e:
        return FileResponse(status_code=404)


def stop_lighthouse_server():
    global lighthouse_server_thread
    if lighthouse_server_thread:
        lighthouse_server_thread.join(10)
        lighthouse_server_thread = None
        bt.logging.info("Lighthouse server stopped")


def start_lighthouse_server():
    try:
        uvicorn.run(app, host="0.0.0.0", port=LIGHTHOUSE_SERVER_PORT)
        bt.logging.success(f"Lighthouse server started on port {LIGHTHOUSE_SERVER_PORT}")
    except Exception as e:
        bt.logging.error(f"Error starting lighthouse server: {e}")
        stop_lighthouse_server()
        raise e



def start_lighthouse_server_thread():
    global lighthouse_server_thread
    try:
        lighthouse_server_thread = threading.Thread(target=start_lighthouse_server, daemon=True)
        lighthouse_server_thread.start()
        bt.logging.info("Lighthouse server started")
    except Exception as e:
        bt.logging.error(f"Error starting lighthouse server: {e}")
        stop_lighthouse_server()
        raise e
