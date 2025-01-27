# WebGenieAI: Validator Documentation

## Compute Requirement
For detailed specifications, please refer to the [Compute Requirement](validator_compute.yml).

## Validator Setup

Follow the steps below to configure and run the Validator.

### Steps

1. **Clone the WebGenieAI Repository:**
   ```bash
   git clone https://github.com/web-genie-ai/web-genie-ai.git
   cd web-genie-ai
   ```

2. **Set Up and Source the `.env` File:**
   - **Create the `.env` File:**
     ```bash
     echo "WANDB_OFF = False" >> .env # Disable Weights & Biases
     echo "WANDB_API_KEY = your_wandb_api_key" >> .env # Your Weights & Biases API key
     echo "WANDB_ENTITY_NAME = your_wandb_entity_name" >> .env # Project name for Weights & Biases
     echo "LLM_API_KEY = your_openai_api_key" >> .env # Your OpenAI API key
     echo "LLM_MODEL_ID = openai_model_id" >> .env # Minimum model ID: gpt-4o
     echo "LLM_MODEL_URL = openai_model_url" >> .env # OpenAI model URL
     echo "LIGHTHOUSE_SERVER_PORT = 5000" >> .env # FastAPI server port for Lighthouse score
     echo "NEURON_EPOCH_LENGTH = 25" >> .env # Default epoch length
     echo "AXON_OFF = True" >> .env # Disable Axon serving
     ```
   - Alternatively, rename `.env.validator.example` and customize it with your values.
   - **Source the `.env` File:**
     ```bash
     source .env
     ```

3. **Run the Validator:**
   - **Using Bash Files:**
     - Install necessary packages:
       ```bash
       bash scripts/install_requirements.sh
       ```
     - Configure your Bittensor wallets and environment variables before proceeding:
       ```bash
       bash scripts/start.sh
       ```
   - **Manually Through Scripts:**
     - Install `pm2` and `uv`:
       ```bash
       npm install pm2 -g
       curl -LsSf https://astral.sh/uv/install.sh | sh
       ```
     - Install the packages in a new terminal:
       ```bash
       uv sync
       ```
     - Install additional dependencies and start the validator:
       ```bash
       npm install -g lighthouse
       wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
       sudo dpkg -i google-chrome-stable_current_amd64.deb
       sudo apt-get install -f
       source .venv/bin/activate
       playwright install-deps
       playwright install
       export PYTHONPATH="."
       pm2 start "uv run neurons/validators/validator.py --netuid [54 | 214] --subtensor.network [finney | test] --wallet.name [coldkey_name] --wallet.hotkey [hotkey_name] --logging.debug --neuron.axon_port [axon_port]" --name webgenie_validator
       ```
   - **Through Auto Update Script:**
     ```bash
     pm2 start --name auto_update auto_update.sh
     ```

---

### Troubleshooting & Support

- **Logs:**
  - For detailed logs, use the following command:
    ```bash
    pm2 logs webgenie_validator
    ```

- **Common Issues:**
  - Missing or incorrect `.env` constants.
  - Unmatched server resource issues.
  - Connectivity problems.

- **Contact Support:** For assistance, please reach out to the WebGenieAI team.

---

Happy Validating!