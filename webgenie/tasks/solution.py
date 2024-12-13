from pydantic import BaseModel, Field

class Solution(BaseModel):
    html: str = Field("", description="The html solution")
    process_time: float = Field(0, description="The time it took to process the solution")
    miner_uid: int = Field(0, description="The uid of the miner that processed the solution")