import sys
import os
from dotenv import load_dotenv, find_dotenv

def init_test():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    load_dotenv(find_dotenv(filename=".env.validator"))
