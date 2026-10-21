from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="Unique session ID for the user conversation.")
    message: str = Field(..., description="The user's message text.")

class ChatResponse(BaseModel):
    thread_id: str
    response: str
