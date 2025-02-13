from pydantic import Field, BaseModel


class DatasetEntry(BaseModel):
    src: str = Field(default="", description="The source of the dataset entry")
    url: str = Field(default="", description="The url of the dataset entry")
    ground_truth_html: str = Field(default="", description="The ground truth html")
    prompt: str = Field(default="", description="The prompt for the text task")
    base64_image: str = Field(default="", description="The base64 encoded image")


class Dataset:
    async def generate_context(self)->DatasetEntry:
        pass
