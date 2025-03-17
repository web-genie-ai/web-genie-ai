import dotenv
dotenv.load_dotenv(".env.validator")

from webgenie.storage import submit_results


submit_results(
    miner_submissions_request={
        "competition": {
            "session_number": 12,
            "competition_type": "seo_competition",
        },
        "submissions": [
            {
                "neuron":{
                    "hotkey": "sdasfasdfd1234567890",
                },
                "score": "0.92"
            },
            {
                "neuron":{
                    "hotkey": "sdasfasdfd1234567890",
                },
                "score": "0.91"
            },
            {
                "neuron":{
                    "hotkey": "sdasfasdfd1234567890",
                },
                "score": "0.90"
            },
            {
                "neuron":{
                    "hotkey": "sdasfasdfd1234567890",
                },
                "score": "0.89"
            },
        ]
    }
)
