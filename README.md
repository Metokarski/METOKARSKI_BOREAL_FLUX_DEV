# Autonomous GPU Image Generator

This project lets you automatically rent a powerful cloud GPU, generate AI images using an interactive prompt, and then automatically shuts the GPU down to save you money.

---

## How to Use (5 Simple Steps)

### 1. Prerequisites
Make sure you have Python 3 installed on your computer.

### 2. Clone the Code
Open your terminal and run this command to download the project:
```bash
git clone https://github.com/Metokarski/METOKARSKI_BOREAL_FLUX_DEV.git
cd kudzueye_boreal_flux_dev
```

### 3. Install Dependencies
Run the following command to install the necessary Python packages:
```bash
pip install -r requirements.txt
```

### 4. Fill Out Your Secrets
This is the most important step. Open the `config/settings.json` file and fill in your secret keys and preferences.

Here is a simple guide for each setting:

| Setting | What It Is (In Simple Terms) | Where to Find It |
| :--- | :--- | :--- |
| `api_key` | Your secret password for Lambda Labs. | On your [Lambda Labs API Keys](https://cloud.lambdalabs.com/api-keys) page. |
| `ssh_key_name` | The **name** you gave your SSH key in the Lambda Labs dashboard. | On your [Lambda Labs SSH Keys](https://cloud.lambdalabs.com/ssh-keys) page. |
| `ssh_private_key_path` | The location of the matching private key file **on your computer**. | Usually `~/.ssh/id_rsa` or wherever you saved your `.pem` file. |
| `hugging_face_token`| Your secret password for Hugging Face to download the AI model. | On your [Hugging Face Access Tokens](https://huggingface.co/settings/tokens) page. |
| `instance_type` | The type of GPU you want to rent. | `gpu_1x_a10` is a good default. |
| `region` | The data center location for your GPU. | `us-tx-1` is a good default. |

**Example `config/settings.json`:**
```json
{
  "api_key": "lam_sk_123abcdeFGHIJKLMN",
  "ssh_key_name": "my-macbook-key",
  "ssh_private_key_path": "~/.ssh/id_rsa",
  "hugging_face_token": "hf_123abcdeFGHIJKLMN",
  "instance_type": "gpu_1x_a10",
  "region": "us-tx-1"
}
```

### 5. Run the App
Once your configuration is saved, run the application from your terminal:
```bash
python run.py
```
The script will now automatically:
1.  Launch a new GPU instance.
2.  Set up the environment.
3.  Start the inference server.

Once it's ready, you will be prompted to enter your image prompt directly in the terminal.

---

### Automatic Cleanup
Don't worry about extra charges. When you are finished and close the script (by typing `quit` or pressing `Ctrl+C`), the application will **automatically terminate the cloud GPU instance** for you.
