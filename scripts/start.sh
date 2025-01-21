#!/bin/bash

# Prompt user for required information
echo "Please enter your neuron details:"
read -p "Coldkey name: " COLDKEY
read -p "Hotkey name: " HOTKEY
read -p "Axon port: " AXON_PORT
echo "If you are going to run validator, PM2 process name will be [webgenie_validator] automatically. If you are going to run miner, please enter the PM2 process name."
read -p "PM2 process name: " PROCESS_NAME

# Prompt for neuron type with validation
while true; do
    read -p "Neuron type (validator/miner): " NEURON_TYPE
    if [[ "$NEURON_TYPE" == "validator" || "$NEURON_TYPE" == "miner" ]]; then
        break
    else
        echo "Invalid neuron type. Please enter either 'validator' or 'miner'"
    fi
done

# Prompt for network type with validation
while true; do
    read -p "Network type (finney/test): " NETWORK
    if [[ "$NETWORK" == "finney" || "$NETWORK" == "test" ]]; then
        break
    else
        echo "Invalid network. Please enter either 'finney' or 'test'"
    fi
done

if [[ "$NEURON_TYPE" == "validator" ]]; then
    PROCESS_NAME="webgenie_validator"
fi

# Confirm the entered values
echo -e "\nYou entered:"
echo "Coldkey: $COLDKEY"
echo "Hotkey: $HOTKEY"
echo "Axon port: $AXON_PORT"
echo "PM2 process name: $PROCESS_NAME"
echo "Neuron type: $NEURON_TYPE"
echo "Network: $NETWORK"

# Ask for confirmation
read -p "Is this correct? (y/n): " CONFIRM

if [[ $CONFIRM != [yY] ]]; then
    echo "Aborted. Please run the script again."
    exit 1
fi

export PYTHONPATH="."
# Set netuid based on network type
NETUID=$([ "$NETWORK" == "finney" ] && echo "54" || echo "214")
if [[ "$NEURON_TYPE" == "validator" ]]; then
    pm2 start "uv run neurons/validators/validator.py --netuid $NETUID --subtensor.network $NETWORK --wallet.name $COLDKEY --wallet.hotkey $HOTKEY --logging.debug --axon.port $AXON_PORT" --name webgenie_validator
    pm2 start --name auto_update scripts/auto_update.sh
else
    pm2 start "uv run neurons/miners/miner.py --netuid $NETUID --subtensor.network $NETWORK --wallet.name $COLDKEY --wallet.hotkey $HOTKEY --logging.debug --axon.port $AXON_PORT" --name $PROCESS_NAME
fi