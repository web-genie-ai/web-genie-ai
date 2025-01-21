import bittensor as bt
import os

# backend api hotkey
API_HOTKEY = "5DXDCYTuPfLqQXbxfvvnarG31SdTDtaubqpQrzjrcMgoP9dp"

# default load time
DEFAULT_LOAD_TIME = 1000

# image task timeout
IMAGE_TASK_TIMEOUT = 72

# text task timeout
TEXT_TASK_TIMEOUT = 72

# reveal time
TASK_REVEAL_TIME = 20

# reveal time out
TASK_REVEAL_TIMEOUT = 20

# lighthouse server port
LIGHTHOUSE_SERVER_PORT = int(os.getenv("LIGHTHOUSE_SERVER_PORT",5000))

# max competition history size
MAX_COMPETETION_HISTORY_SIZE = 30

# max synthetic task size
MAX_SYNTHETIC_TASK_SIZE = 30

# max debug image string length
MAX_DEBUG_IMAGE_STRING_LENGTH = 20

# place holder image url
PLACE_HOLDER_IMAGE_URL = "https://picsum.photos/seed/picsum/800/600"

# python command
PYTHON_CMD = "python"

# screenshot script path
SCREENSHOT_SCRIPT_PATH = "webgenie/rewards/visual_reward/metrics/screenshot_single.py"

# max page load time
GROUND_TRUTH_HTML_LOAD_TIME = 20000

# miner html load time
CHROME_HTML_LOAD_TIME = 60000

# miner html load time
MINER_HTML_LOAD_TIME = 2000

# max miner html length
MAX_MINER_HTML_LEN = 1000000

# work dir
WORK_DIR = "work"

# lighthouse server work dir
LIGHTHOUSE_SERVER_WORK_DIR = f"{WORK_DIR}/lighthouse_server_work"

# html extension
HTML_EXTENSION = ".html"

# image extension
IMAGE_EXTENSION = ".png"

# max count of validators
MAX_COUNT_VALIDATORS = 1

# block in seconds
BLOCK_IN_SECONDS = 12

# tempo blocks
TEMPO_BLOCKS = 360

# session window blocks
SESSION_WINDOW_BLOCKS = TEMPO_BLOCKS * 3

# querying window blocks
QUERING_WINDOW_BLOCKS = 10

# weight setting window blocks
WEIGHT_SETTING_WINDOW_BLOCKS = 50 # 50 blocks = 10 minutes

# llm model id
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")

# llm api key
LLM_API_KEY = os.getenv("LLM_API_KEY")

# llm model url
LLM_MODEL_URL = os.getenv("LLM_MODEL_URL")

# wandb api key
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# wandb project name
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")

# wandb entity name
WANDB_ENTITY_NAME = os.getenv("WANDB_ENTITY_NAME")

# vpermit tao limit
#VPERMIT_TAO_LIMIT = bt.Balance(float(os.getenv("VPERMIT_TAO_LIMIT", 4096)))
VPERMIT_TAO_LIMIT = float(os.getenv("VPERMIT_TAO_LIMIT", 4096))

# axon off
AXON_OFF = os.getenv("AXON_OFF", "False").lower() == "true"

