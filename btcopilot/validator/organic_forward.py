from typing import Callable
from functools import partial
from typing import Any, AsyncGenerator
from starlette.types import Send

import bittensor as bt
from btcopilot.protocol import BtCopilotSynapse

async def forward_organic_synapse(self, synapse: BtCopilotSynapse)->BtCopilotSynapse:
    async def forward_miner(synapse: BtCopilotSynapse, send: Send):
        bt.logging.info(f"Send Synapse to miner: {synapse}")  
        async def handle_miner_response(responses):
            for resp in responses:
                async for chunk in resp:
                    if isinstance(chunk, str):
                        bt.logging.info(f"Chunk: {chunk}")
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
        try:
            async with bt.dendrite(wallet=self.wallet) as dendrite:
                bt.logging.info(f"Dendrite: {dendrite}")
                responses = await dendrite(
                    axons=[axon],
                    synapse=synapse,
                    deserialize=False,
                    timeout=synapse.timeout,
                    streaming=True,
                )
        except Exception as e:
            bt.logging.error(f"[forward_organic_synapse] Error querying dendrite: {e}")
        return await handle_miner_response(responses)
    bt.logging.info(f"forward_organic_synapse: {synapse}")
    send_external_response = partial(forward_miner, synapse)
    return synapse.create_streaming_response(send_external_response)
    