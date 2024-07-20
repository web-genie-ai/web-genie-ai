# Created by Brendon R.

import traceback
import bittensor as bt
from neurons import protocol

async def forward_codegen(self, synapse: protocol.CodeGenObj ) -> protocol.CodeGenObj:
        try:
            synapse.output = protocol.CodeGenObj(
                metadata=protocol.DiscoveryMetadata(
                    network=self.config.network,
                ),
            )
            bt.logging.info(f"Serving miner discovery output: {synapse.output}")
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            synapse.output = None
        return synapse

async def forward_codegen_blacklist(self, synapse: protocol.CodeGenObj) -> typing.Tuple[bool, str]:
        return blacklist.forward_codegen_blacklist(self, synapse=synapse)

async def forward_codegen_priority(self, synapse: protocol.Discovery) -> float:
        return self.forward_codegen_priority(synapse=synapse)