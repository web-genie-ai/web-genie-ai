import bittensor as bt
import os
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer

from webgenie.constants import (
    LIGHTHOUSE_SERVER_WORK_DIR,
    LIGHTHOUSE_SERVER_PORT,
)


httpd = None


class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.directory = LIGHTHOUSE_SERVER_WORK_DIR
        super().__init__(*args, **kwargs)


def start_lighthouse_server():
    global httpd
    try:
        httpd = HTTPServer(('localhost', LIGHTHOUSE_SERVER_PORT), CustomHTTPRequestHandler)
        httpd.serve_forever()
        bt.logging.success(f"Lighthouse server started on port {LIGHTHOUSE_SERVER_PORT}")
    except Exception as e:
        bt.logging.error(f"Error starting lighthouse server: {e}")
        httpd.shutdown()
        httpd.server_close()
        httpd = None
        bt.logging.info("Server stopped")


def start_lighthouse_server_thread():
    global httpd
    try:
        lighthouse_server_thread = threading.Thread(target=start_lighthouse_server, daemon=True)
        lighthouse_server_thread.start()
        bt.logging.info("Lighthouse server started")
    except Exception as e:
        bt.logging.error(f"Error starting lighthouse server: {e}")
        httpd.shutdown()
        httpd.server_close()
        bt.logging.info("Server stopped")


def stop_lighthouse_server():
    global httpd
    if httpd:
        httpd.shutdown()
        httpd.server_close()
        httpd = None
        bt.logging.info("Server stopped")
