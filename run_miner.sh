export PYTHONPATH=.
#--axon.port 5555
python3.12 neurons/miners/miner.py --netuid 214 --subtensor.network test --wallet.name s-miner --wallet.hotkey miner1 --logging.debug --axon.port 8090

