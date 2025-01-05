export PYTHONPATH=.

pm2 start neurons/validators/validator.py --name "webgenie_validator" --interpreter python -- --netuid 214 --subtensor.network test --wallet.name sc-val1 --wallet.hotkey sh-val1 --logging.debug --neuron.axon_port 8091