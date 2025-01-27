# WebGenieAI: Validator Documentation

## Compute Requirement
For detailed specifications, please refer to the [Compute Requirement](validator_compute.yml).

## Validator Setup

Follow the steps below to configure and run the Validator.

### Steps

1. **Clone the WebGenieAI Repository**
   ```bash
   git clone https://github.com/web-genie-ai/web-genie-ai.git
   cd web-genie-ai
   ```

2. **Configure the Environment Variables**
   - **Create the `.env` | `.env.validator` File:**
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
   - Alternatively, rename `.env.validator.example` and customize it with your environment variables.

3. **Run the Validator**
   - **Using Bash Files**
     - Install necessary packages
       ```bash
       bash scripts/install_requirements.sh
       ```
     - Configure your Bittensor wallets
     - Start the validator
       ```bash
       bash scripts/start.sh
       ```

---

### Troubleshooting & Support

- **Logs**
  - For detailed logs, use the following command
    ```bash
    pm2 logs webgenie_validator
    ```

- **Common Issues**
  - Missing or incorrect `.env` variables.
  - If nodejs version is below v20, you may encounter issues with lighthouse score.
  - If your lighthouse fastapi server(default port 5000) is not running, for example port binding error, you may encounter issues with lighthouse score.

- **Contact Support**
  - For assistance, please reach out to the WebGenieAI team.<br />
    Email: support@webgenieai.co | sangar@webgenieai.co <br />
    Discord: https://discord.com/channels/799672011265015819/1316415472563916841

---

Happy Validating!