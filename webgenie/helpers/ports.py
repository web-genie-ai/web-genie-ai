import os

def kill_process_on_port(port):
    os.system(f"sudo kill -9 $(sudo lsof -t -i :{port})")