from typing import Callable
from functools import partial
from typing import Any, AsyncGenerator
from starlette.types import Send

import bittensor as bt
from btcopilot.protocol import BtCopilotSynapse

def forward_organic_synapse(self, synapse: BtCopilotSynapse)->BtCopilotSynapse:
    async def forward_miner(synapse, send: Send):
        async def handle_miner_response(responses):
            for resp in responses:
                async for chunk in resp:
                    if isinstance(chunk, str):
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )

        axon = self.metagraph.axons[1]
        responses = self.dendrite.query(
            axons=[axon],
            synapse=synapse,
            deserialize=False,
            timeout=synapse.timeout,
            streaming=True,
        )
        return await handle_miner_response(responses)
    
    send_external_response = partial(forward_miner, synapse)
    return synapse.create_streaming_response(send_external_response)
    