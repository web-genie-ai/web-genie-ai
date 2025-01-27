# WebGenieAI: Miner Documentation

## Compute Requirement
For detailed specifications, please refer to the [Compute Requirement](miner_compute.yml).

## Miner Setup

Follow the steps below to configure and run the Miner. You can use either a closed-source AI service like OpenAI or your customized model.

### Setup Instructions

1. **Clone the WebGenieAI Repository**
   ```bash
   git clone https://github.com/web-genie-ai/web-genie-ai.git
   cd web-genie-ai
   ```

2. **Configure the Environment Variables**
   - **Create the `.env`| `.env.miner` File**
     ```bash
     echo "WANDB_OFF = False" >> .env # Turn off wandb.
     echo "WANDB_API_KEY = your_wandb_api_key" >> .env # Your wandb api key, for example: sk-proj-1234567890
     echo "WANDB_ENTITY_NAME = your_wandb_entity_name" >> .env # The name of the project where you are sending the new run.
     echo "LLM_API_KEY = your_openai_api_key" >> .env # Your openai api key, for example: sk-proj-1234567890
     echo "LLM_MODEL_ID = openai_model_id" >> .env # Minimum model ID: gpt-4o when using OpenAI for mining; not required for custom models
     echo "LLM_MODEL_URL = openai_model_url" >> .env # OpenAI model URL when using OpenAI for mining; not required for custom models
     echo "HF_TOKEN = your_huggingface_token" >> .env # Your Hugging Face token, e.g., hf_1234567890; not required when using OpenAI for mining
     ```
   - Alternatively, rename `.env.miner.example` and customize it with your environment variables.

3. **Execute the Miner**
   - **Using Bash Scripts**
     - Install necessary packages
       ```bash
       bash scripts/install_requirements.sh
       ```
     - Configure your Bittensor wallets
     - Start the miner
       ```bash
       bash scripts/start.sh
       ```

---

### Troubleshooting & Support

- **Logs**
  - For detailed logs, use the following command
    ```bash
    pm2 logs webgenie_miner
    ```

- **Common Issues:**
  - Missing or incorrect `.env` variables.

- **Contact Support**
  - Email: support@webgenieai.co | sangar@webgenieai.co
  - Discord: https://discord.com/channels/799672011265015819/1316415472563916841

---

Happy Mining!