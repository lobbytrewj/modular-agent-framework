from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


# What work needs to be done, and which agent is responsible for doing it
class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    assigned_agent: str
    input_data: dict[str, Any] = Field(default_factory=dict)


# The final outcome of running a Task, success or failure
class RunResult(BaseModel):
    task_id: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
