import bittensor as bt
import os
import time
import psutil


def install_sudo():
    # Check if sudo is already installed
    if os.system("sudo --version") == 0:
        bt.logging.info("✅ Sudo is already installed.")
        return
    
    bt.logging.info("❌ Sudo is not installed. Installing now...")

    # Detect Linux distribution and install sudo
    if os.path.exists("/etc/debian_version"):
        package_manager = "apt"
        install_cmd = "apt update && apt install -y sudo"
    elif os.path.exists("/etc/redhat-release"):
        package_manager = "yum" if os.path.exists("/usr/bin/yum") else "dnf"
        install_cmd = f"{package_manager} install -y sudo"
    else:
        bt.logging.error("❌ Unsupported Linux distribution. Cannot install sudo.")
        return
    
    # Run the installation command as root
    bt.logging.info(f"🔄 Installing sudo using {package_manager}...")
    os.system(f"su -c '{install_cmd}'")

    # Verify installation
    if os.system("sudo --version") == 0:
        bt.logging.info("✅ Sudo has been successfully installed.")
    else:
        bt.logging.error("❌ Failed to install sudo.")

# Run the function
install_sudo()

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

