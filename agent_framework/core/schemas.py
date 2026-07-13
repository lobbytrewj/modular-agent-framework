from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

# info when a message is sent or recieved
class AgentMessage(BaseModel):
    # Creates unique id
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    recipient: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
