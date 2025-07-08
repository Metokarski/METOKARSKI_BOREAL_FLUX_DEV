import os
import json
import time
import paramiko
import requests
from lambdalabs_api.util import launch_inst

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
    print(f"--- Provisioning instance at {ip_address} ---")
    
    # Expand the user's home directory (e.g., '~')
    ssh_key_path = os.path.expanduser(ssh_private_key_path)

    if not os.path.exists(ssh_key_path):
        raise Exception(f"SSH private key not found at the specified path: {ssh_key_path}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    for i in range(10):
        try:
            print(f"Connecting to {ip_address} (attempt {i+1}/10)...")
            ssh.connect(ip_address, username='ubuntu', key_filename=ssh_key_path, timeout=20)
            print("SSH connection successful!")
            break
        except Exception as e:
            print(f"Connection failed: {e}. Retrying in 15 seconds...")
            time.sleep(15)
    else:
        raise Exception("Could not establish SSH connection.")

    repo_url = "https://github.com/Metokarski/METOKARSKI_BOREAL_FLUX_DEV_V2.git"
    repo_name = "METOKARSKI_BOREAL_FLUX_DEV_V2"
    
    # Securely inject the Hugging Face token as an environment variable
    start_server_command = (
        f"export HUGGING_FACE_TOKEN='{hugging_face_token}'; "
        "nohup ~/.local/bin/uvicorn inference_server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
    )

    commands = [
        "sudo apt-get update",
        "sudo apt-get install -y git python3-pip",
        f"git clone {repo_url}",
        f"cd {repo_name} && pip3 install -r requirements.txt",
        f"cd {repo_name} && {start_server_command}"
    ]

    for command in commands:
        print(f"Executing: {command}")
        stdin, stdout, stderr = ssh.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_message = stderr.read().decode()
            raise Exception(f"Error executing command: {command}\n{error_message}")
        print(stdout.read().decode())

    ssh.close()
    print("--- Provisioning complete ---")

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

    def __enter__(self):
        """Called when entering the 'with' block."""
        self.instance_id, self.ip_address = launch_and_provision()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting the 'with' block."""
        if self.instance_id:
            terminate_instance(self.instance_id)
