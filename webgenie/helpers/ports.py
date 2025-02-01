import bittensor as bt
import os
import time
import psutil


def kill_process_on_port(port):
    try:
        cmd = f"sudo kill -9 $(sudo lsof -t -i :{port})"
        os.system(cmd)
        time.sleep(1)
    except Exception as e:
        bt.logging.error(f"Error killing process on port {port}: {e}")
        raise Exception(f"Error killing process on port {port}: {e}")

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    raise Exception(f"Error killing process on port {port}: {e}")
        except Exception as e:
            bt.logging.error(f"Error killing process on port {port}: {e}")
            raise Exception(f"Error killing process on port {port}: {e}")

