import bittensor as bt
import os
import requests

from webgenie.constants import API_TOKEN, DASHBOARD_BACKEND_URL


def submit_results(miner_submissions_request: dict):
    try:
        url = f"{DASHBOARD_BACKEND_URL}/api/submit_results"
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=miner_submissions_request, headers=headers)
        if response.status_code != 200:
            bt.logging.error(f"Error submitting results: {response.status_code} {response.text}")
            return
        response_json = response.json()
        if response_json.get("success"):
            bt.logging.success(f"Results submitted successfully")
        else:
            bt.logging.error(f"Error submitting results")
    except Exception as e:
        bt.logging.error(f"Error submitting results: {e}")