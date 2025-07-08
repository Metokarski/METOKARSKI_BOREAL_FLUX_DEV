import os
import json
import time
import paramiko
import requests
import subprocess
from core.logger import get_logger
from lambdalabs_api.util import launch_inst

log = get_logger(__name__)
CONFIG_PATH = 'config/settings.json'

def get_config():
    """Loads and validates configuration from the settings file."""
    if not os.path.exists(CONFIG_PATH):
        raise Exception(f"Configuration file not found at {CONFIG_PATH}")
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    required_keys = ["api_key", "ssh_key_name", "ssh_private_key_path", "hugging_face_token", "instance_type", "region"]
    for key in required_keys:
        if key not in config or not config[key] or "YOUR_" in str(config[key]) or "your_key" in str(config[key]):
            raise Exception(f"Configuration error: Please set a valid value for '{key}' in {CONFIG_PATH}")

    return config

def get_instance_details(instance_id, api_key):
    """Gets the latest details for a specific instance."""
    url = f"https://cloud.lambdalabs.com/api/v1/instances/{instance_id}"
    headers = {"Authorization": f"Basic {api_key}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['data']

def _provision_instance(ip_address, ssh_private_key_path, hugging_face_token):
    """Connects to the instance via SSH and provisions it."""
    log.info(f"--- Provisioning instance at {ip_address} ---")
    
    ssh_key_path = os.path.expanduser(ssh_private_key_path)
    if not os.path.exists(ssh_key_path):
        log.error(f"SSH private key not found at {ssh_key_path}")
        raise Exception(f"SSH private key not found at {ssh_key_path}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        for i in range(10):
            try:
                log.info(f"Connecting to {ip_address} (attempt {i+1}/10)...")
                ssh.connect(ip_address, username='ubuntu', key_filename=ssh_key_path, timeout=20)
                log.info("SSH connection successful!")
                break
            except Exception as e:
                log.warning(f"Connection attempt {i+1} failed: {e}. Retrying in 15 seconds...")
                time.sleep(15)
        else:
            log.error("Could not establish SSH connection after multiple retries.")
            raise Exception("Could not establish SSH connection.")

        repo_url = "https://github.com/Metokarski/METOKARSKI_BOREAL_FLUX_DEV.git"
        repo_name = "METOKARSKI_BOREAL_FLUX_DEV"
        provision_log_file = "provisioning.log"
        server_log_file = "server.log"

        commands = [
            f"echo '--- Starting Provisioning ---' > {provision_log_file}",
            f"sudo apt-get update >> {provision_log_file} 2>&1",
            f"sudo apt-get install -y git python3-pip >> {provision_log_file} 2>&1",
            f"git clone {repo_url} >> {provision_log_file} 2>&1",
            f"cd {repo_name} && pip3 install -r requirements.txt >> ../{provision_log_file} 2>&1",
            f"cd {repo_name} && export HUGGING_FACE_TOKEN='{hugging_face_token}'; nohup /home/ubuntu/.local/bin/uvicorn inference_server:app --host 0.0.0.0 --port 8000 > {server_log_file} 2>&1 &"
        ]

        for command in commands:
            log.info(f"Executing remote command: {command}")
            stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
            exit_status = stdout.channel.recv_exit_status()
            
            stdout_str = stdout.read().decode()
            stderr_str = stderr.read().decode()

            if stdout_str:
                log.debug(f"STDOUT: {stdout_str}")
            if stderr_str:
                log.warning(f"STDERR: {stderr_str}")

            if exit_status != 0:
                log.error(f"Command failed with exit status {exit_status}: {command}")
                raise Exception(f"Error executing command: {command}\nSTDERR: {stderr_str}")
            log.info(f"Command finished successfully: {command}")

    except Exception as e:
        log.error(f"An error occurred during provisioning: {e}", exc_info=True)
        raise
    finally:
        ssh.close()
        log.info("SSH connection closed.")
    log.info("--- Provisioning complete ---")

def launch_and_provision():
    """
    Main function to launch, monitor, and provision the instance.
    Returns the IP address of the ready server.
    """
    config = get_config()
    api_key = config['api_key']
    ssh_key_name = config['ssh_key_name']
    ssh_private_key_path = config['ssh_private_key_path']
    hugging_face_token = config['hugging_face_token']
    instance_type = config['instance_type']
    region = config['region']

    print("--- Starting instance launch ---")
    instance_data = launch_inst(instance_type, region, ssh_key_name)
    instance_id = instance_data['id']
    print(f"Instance {instance_id} is launching...")

    ip_address = None
    while True:
        details = get_instance_details(instance_id, api_key)
        if details.get('ip') and details['status'] == 'active':
            ip_address = details['ip']
            print(f"Instance {instance_id} is active with IP: {ip_address}")
            break
        print(f"Instance status: {details['status']}. Waiting...")
        time.sleep(20)

    _provision_instance(ip_address, ssh_private_key_path, hugging_face_token)
    
    # Give the server a moment to start up
    print("Waiting for inference server to initialize...")
    time.sleep(15)
    
    return instance_id, ip_address

def terminate_instance(instance_id: str):
    """Terminates a Lambda Labs instance."""
    config = get_config()
    api_key = config['api_key']
    
    print(f"\n--- Terminating instance {instance_id} ---")
    url = "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate"
    headers = {"Authorization": f"Basic {api_key}"}
    payload = {"instance_ids": [instance_id]}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Instance {instance_id} termination request successful.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error terminating instance {instance_id}: {e}")
        # Decide if you want to raise the exception or just log it
        # For a cleanup function, it's often better to just log and continue
        pass

class ManagedGPU:
    """A context manager to handle the lifecycle of a GPU instance."""
    def __init__(self):
        self.instance_id = None
        self.ip_address = None
        self.ssh_private_key_path = None

    def __enter__(self):
        """Called when entering the 'with' block."""
        try:
            config = get_config()
            self.ssh_private_key_path = os.path.expanduser(config['ssh_private_key_path'])
            self.instance_id, self.ip_address = launch_and_provision()
            return self
        except Exception as e:
            log.error("Failed to enter ManagedGPU context.", exc_info=True)
            # Ensure __exit__ is still called for cleanup
            self.__exit__(type(e), e, e.__traceback__)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting the 'with' block."""
        if exc_type:
            log.error(f"Exiting ManagedGPU context due to an exception: {exc_val}", exc_info=(exc_type, exc_val, exc_tb))

        if self.ip_address:
            log.info("--- Downloading logs from instance ---")
            repo_name = "METOKARSKI_BOREAL_FLUX_DEV"
            # Note: The log files are in the home directory, not inside the repo folder
            logs_to_download = {"provisioning.log": "provisioning.log", "server.log": f"{repo_name}/server.log"}
            
            for local_name, remote_path in logs_to_download.items():
                try:
                    scp_command = [
                        "scp",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null",
                        "-i", self.ssh_private_key_path,
                        f"ubuntu@{self.ip_address}:{remote_path}",
                        local_name
                    ]
                    log.info(f"Executing: {' '.join(scp_command)}")
                    result = subprocess.run(scp_command, check=True, capture_output=True, text=True, timeout=60)
                    log.info(f"Successfully downloaded {remote_path} to {local_name}")
                    if result.stdout:
                        log.debug(f"SCP STDOUT: {result.stdout}")
                    if result.stderr:
                        log.debug(f"SCP STDERR: {result.stderr}")
                except subprocess.CalledProcessError as e:
                    log.error(f"Failed to download {remote_path}. SCP command failed with exit code {e.returncode}.")
                    log.error(f"SCP Stderr: {e.stderr}")
                except FileNotFoundError:
                    log.warning(f"Could not download {remote_path}, file not found on remote instance.")
                except Exception as e:
                    log.error(f"An unexpected error occurred while downloading {remote_path}: {e}", exc_info=True)

        if self.instance_id:
            terminate_instance(self.instance_id)
        log.info("--- ManagedGPU context exited ---")
