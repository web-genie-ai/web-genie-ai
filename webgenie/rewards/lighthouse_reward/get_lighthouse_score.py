import bittensor as bt
import json
import subprocess
import threading
import time
import uuid
import os

from http.server import SimpleHTTPRequestHandler, HTTPServer
from typing import List, Dict

from webgenie.constants import (
    LIGHTHOUSE_SERVER_PORT, 
    LIGHTHOUSE_SERVER_WORK_DIR,
)

def get_lighthouse_score(htmls: List[str]) -> List[Dict[str, float]]:
    def get_lighthouse_score_from_subprocess(url):
        bt.logging.info(f"Getting lighthouse score from {url}...")
        try:
            result = subprocess.run(
                ['lighthouse', url, '--output=json', '--quiet', '--chrome-flags="--headless --no-sandbox"'],
                capture_output=True, text=True, timeout=180
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

    bt.logging.info(f"Getting lighthouse scores from localhost:{LIGHTHOUSE_SERVER_PORT}...")
    scores = []
    
    for i in range(len(htmls)):
        
        file_name = f"{uuid.uuid4()}.html"

        with open(f"{LIGHTHOUSE_SERVER_WORK_DIR}/{file_name}", "w") as f:
            f.write(htmls[i])

        url = f"http://localhost:{LIGHTHOUSE_SERVER_PORT}/{file_name}"
        scores.append(get_lighthouse_score_from_subprocess(url))
        
        os.remove(f"{LIGHTHOUSE_SERVER_WORK_DIR}/{file_name}")

    return scores