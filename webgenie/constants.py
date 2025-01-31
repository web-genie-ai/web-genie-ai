import bittensor as bt
import os

# Change this value when updating your code base.
# Define the version of the webgenie.
__VERSION__ = "1.0.10" # version

SPEC_VERSION = (
    (1000 * int(__VERSION__.split(".")[0]))
    + (10 * int(__VERSION__.split(".")[1]))
    + (1 * int(__VERSION__.split(".")[2]))
)

__STATE_VERSION__ = "1.0.0" # state version

API_HOTKEY = "5DXDCYTuPfLqQXbxfvvnarG31SdTDtaubqpQrzjrcMgoP9dp" # backend api hotkey

IMAGE_TASK_TIMEOUT = 72 # image task timeout

TEXT_TASK_TIMEOUT = 72 # text task timeout

TASK_REVEAL_TIME = 20 # reveal time

TASK_REVEAL_TIMEOUT = 20 # reveal time out

LIGHTHOUSE_SERVER_PORT = int(os.getenv("LIGHTHOUSE_SERVER_PORT",5000)) # lighthouse server port

MAX_COMPETETION_HISTORY_SIZE = 10 # max competition history size

MAX_SYNTHETIC_TASK_SIZE = 10 # max synthetic task size

MAX_DEBUG_IMAGE_STRING_LENGTH = 20 # max debug image string length

PLACE_HOLDER_IMAGE_URL = "https://picsum.photos/seed/picsum/800/600" # place holder image url

DEFAULT_LOAD_TIME = 1000 # default load time

GROUND_TRUTH_HTML_LOAD_TIME = 20000 # max page load time

CHROME_HTML_LOAD_TIME = 60000 # miner html load time

JAVASCRIPT_RUNNING_TIME = 1000 # javascript running time

MINER_HTML_LOAD_TIME = 2000 # miner html load time

MAX_MINER_HTML_LEN = 1000000 # max miner html length

WORK_DIR = "work" # work dir

LIGHTHOUSE_SERVER_WORK_DIR = f"{WORK_DIR}/lighthouse_server_work" # lighthouse server work dir

HTML_EXTENSION = ".html" # html extension

IMAGE_EXTENSION = ".png" # image extension

MAX_COUNT_VALIDATORS = 1 # max count of validators

BLOCK_IN_SECONDS = 12 # block in seconds

TEMPO_BLOCKS = 360 # tempo blocks

SESSION_WINDOW_BLOCKS = TEMPO_BLOCKS * 5 # session window blocks

CONSIDERING_SESSION_COUNTS = 8

QUERING_WINDOW_BLOCKS = 10

WEIGHT_SETTING_WINDOW_BLOCKS = 50 # 50 blocks = 10 minutes

LLM_MODEL_ID = os.getenv("LLM_MODEL_ID") # llm model id

LLM_API_KEY = os.getenv("LLM_API_KEY") # llm api key

LLM_MODEL_URL = os.getenv("LLM_MODEL_URL") # llm model url

WANDB_OFF = os.getenv("WANDB_OFF", "False").lower() == "true" # wandb off

WANDB_API_KEY = os.getenv("WANDB_API_KEY") # wandb api key

WANDB_PROJECT_NAME = f"webgenie" # wandb project name

WANDB_ENTITY_NAME = os.getenv("WANDB_ENTITY_NAME") # wandb entity name

VPERMIT_TAO_LIMIT = 1000 # vpermit tao limit

AXON_OFF = os.getenv("AXON_OFF", "False").lower() == "true" # axon off

NEURON_EPOCH_LENGTH = int(os.getenv("NEURON_EPOCH_LENGTH", 25)) # neuron epoch length

