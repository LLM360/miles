from pydantic import BaseModel, ConfigDict


class FtBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
