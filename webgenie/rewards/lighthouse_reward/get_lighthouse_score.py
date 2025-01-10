import bittensor as bt
import json
import subprocess
import threading
import time

from http.server import SimpleHTTPRequestHandler, HTTPServer
from typing import List, Dict

from webgenie.constants import LIGHTHOUSE_SERVER_PORT


def get_lighthouse_score(htmls: List[str]) -> List[Dict[str, float]]:
    class CustomHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def do_GET(self):
            # Add CORS headers
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            if self.path == '/favicon.ico':
                self.send_response(200)
                self.send_header('Content-type', 'image/x-icon')
                self.end_headers()
                self.wfile.write(b'')  # send a blank byte or actual icon data
            elif self.path == '/robots.txt':
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'User-agent: *\nDisallow: /')  # Example content
            elif self.path.startswith('/lighthouse_score'):
                print(f"Serving HTML {self.path}")
                html_index = int(self.path.split('/')[-1])
                print(f"HTML index: {html_index}")
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(htmls[html_index].encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")

        def do_OPTIONS(self):
            # Handle preflight requests
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS') 
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

    def run_server(port=8000):
        server_address = ('', port)
        httpd = HTTPServer(server_address, CustomHandler)
        bt.logging.info(f"Starting server on port {port}...")
        httpd.serve_forever()

    port = LIGHTHOUSE_SERVER_PORT
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()
    
    def get_lighthouse_score_from_subprocess(url):
        try:
            result = subprocess.run(
                ['lighthouse', url, '--output=json', '--quiet', '--chrome-flags="--headless --no-sandbox"'],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                lighthouse_report = json.loads(result.stdout)
                scores = {
                    'performance': lighthouse_report['categories']['performance']['score'],
                    'accessibility': lighthouse_report['categories']['accessibility']['score'],
                    'best-practices': lighthouse_report['categories']['best-practices']['score'],
                    'seo': lighthouse_report['categories']['seo']['score']
                }
                return scores
            else:
                bt.logging.error(f"Error running Lighthouse: {result.stderr}")
        except Exception as e:
            bt.logging.error(f"Error running Lighthouse: {e}")
            return {
                'performance': 0,
                'accessibility': 0,
                'best-practices': 0,
                'seo': 0
            }

    time.sleep(1)  # Give the server time to start
    scores = []
    for i in range(len(htmls)):
        url = f"http://localhost:{port}/lighthouse_score/{i}"
        scores.append(get_lighthouse_score_from_subprocess(url))
    
    server_thread.join(timeout=10)
    if server_thread.is_alive():
        bt.logging.info("Server did not shut down properly.")
    else:
        bt.logging.info("Server stopped successfully.")
    
    return scores

    
    