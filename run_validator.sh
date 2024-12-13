export PYTHONASYNCIODEBUG=1
export PYTHONPATH=.
python neurons/validators/validator.py --netuid 214 --subtensor.network test --wallet.name sc-val1 --wallet.hotkey sh-val1 --logging.debug  --neuron.axon_port 8091
