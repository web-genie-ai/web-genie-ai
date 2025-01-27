# WebGenieAI: Miner Documentation

## Compute Requirement
For detailed specifications, please refer to the [Compute Requirement](miner_compute.yml).

## Miner Setup

Follow the steps below to configure and run the Miner. You can use either a closed-source AI service like OpenAI or your customized model.

### Setup Instructions

1. **Clone the WebGenieAI Repository:**
   ```bash
   git clone https://github.com/web-genie-ai/web-genie-ai.git
   cd web-genie-ai
   ```

2. **Configure the Environment:**
   - **Create the `.env` File:**
     ```bash
     echo "LLM_API_KEY = your_openai_api_key" >> .env # Your OpenAI API key when using OpenAI for mining; not required for custom models
     echo "LLM_MODEL_ID = openai_model_id" >> .env # Minimum model ID: gpt-4o when using OpenAI for mining; not required for custom models
     echo "LLM_MODEL_URL = openai_model_url" >> .env # OpenAI model URL when using OpenAI for mining; not required for custom models
     echo "HF_TOKEN = your_huggingface_token" >> .env # Your Hugging Face token, e.g., hf_1234567890; not required when using OpenAI for mining
     ```
   - Alternatively, rename `.env.miner.example` and customize it with your values.
   - **Source the `.env` File:**
     ```bash
     source .env
     ```

3. **Execute the Miner:**
   - **Using Bash Scripts:**
     - Install necessary packages:
       ```bash
       bash scripts/install_requirements.sh
       ```
     - Configure your Bittensor wallets and environment variables before proceeding:
       ```bash
       bash scripts/start.sh
       ```
   - **Manual Execution:**
     - Install `pm2` and `uv`:
       ```bash
       npm install pm2 -g
       curl -LsSf https://astral.sh/uv/install.sh | sh
       ```
     - Install the packages in a new terminal:
       ```bash
       uv sync
       ```
     - Start the miner:
       ```bash
       export PYTHONPATH="."
       pm2 start "uv run neurons/miners/miner.py --netuid [54 | 214] --subtensor.network [finney | test] --wallet.name [coldkey_name] --wallet.hotkey [hotkey_name] --logging.debug --axon.port [axon_port]" --name webgenie_miner
       ```

---

### Troubleshooting & Support

- **Logs:**
  - For detailed logs, use the following command:
    ```bash
    pm2 logs webgenie_miner
    ```

- **Common Issues:**
  - Missing or incorrect `.env` constants.
  - Unmatched server resource issues.
  - Connectivity problems.

- **Contact Support:** For assistance, please reach out to the WebGenieAI team.

---

Happy Mining!