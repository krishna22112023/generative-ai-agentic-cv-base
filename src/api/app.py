"""
FastAPI application for AgenticVision.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import asyncio
import contextlib
from typing import AsyncGenerator, Dict, List, Any

from src.graph import build_graph
from src.config import TEAM_MEMBERS
from src.service.workflow_service import run_agent_workflow
from src.utils.versioning import ensure_dataset_async

# Configure logging
logger = logging.getLogger(__name__)

# Modify the FastAPI app to use a lifespan context
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async def cleanup_expired_states():
        while True:
            await asyncio.sleep(300)  # Run every 5 minutes
            current_time = asyncio.get_event_loop().time()
            expired_keys = []
            
            for workflow_id, state in workflow_states.items():
                # Expire states after 30 minutes
                if current_time - state["timestamp"] > 1800:
                    expired_keys.append(workflow_id)
            
            for key in expired_keys:
                logger.info(f"Removing expired workflow state: {key}")
                workflow_states.pop(key, None)
    
    # Start the cleanup task
    cleanup_task = asyncio.create_task(cleanup_expired_states())
    try:
        yield
    finally:
        # Cancel the cleanup task when the app shuts down
        cleanup_task.cancel()
        await cleanup_task

# Create FastAPI app
app = FastAPI(
    title="AgenticVision API",
    description="API for AgenticVision, LangGraph-based agent workflow",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create the graph
graph = build_graph()

# Store for interrupted workflow state
workflow_states = {}


class ContentItem(BaseModel):
    type: str = Field(..., description="The type of content (text, image, etc.)")
    text: Optional[str] = Field(None, description="The text content if type is 'text'")
    image_url: Optional[str] = Field(
        None, description="The image URL if type is 'image'"
    )


class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="The role of the message sender (user or assistant)"
    )
    content: Union[str, List[ContentItem]] = Field(
        ...,
        description="The content of the message, either a string or a list of content items",
    )


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="The conversation history")
    debug: Optional[bool] = Field(False, description="Whether to enable debug logging")
    deep_thinking_mode: Optional[bool] = Field(
        False, description="Whether to enable deep thinking mode"
    )
    search_before_planning: Optional[bool] = Field(
        False, description="Whether to search before planning"
    )
    team_members: Optional[List[str]] = Field(
        default_factory=list,
        description="List of team members associated with this request",
    )
    dataset: Optional[Dict[str, Optional[str]]] = Field(
        None,
        description="Dataset context containing project_id, version_id, bucket_name, and project_name",
    )
   #workflow_id: str = Field(..., description="ID of the workflow to resume")
   #user_input: str = Field(..., description="User input to continue the workflow")


@app.post("/api/chat/stream")
async def chat_endpoint(request: ChatRequest, req: Request):
    """
    Chat endpoint for LangGraph invoke.

    Args:
        request: The chat request
        req: The FastAPI request object for connection state checking

    Returns:
        The streamed response
    """
    try:
        # Convert Pydantic models to dictionaries and normalize content format
        messages = []
        for msg in request.messages:
            message_dict = {"role": msg.role}

            # Handle both string content and list of content items
            if isinstance(msg.content, str):
                message_dict["content"] = msg.content
            else:
                # For content as a list, convert to the format expected by the workflow
                content_items = []
                for item in msg.content:
                    if item.type == "text" and item.text:
                        content_items.append({"type": "text", "text": item.text})
                    elif item.type == "image" and item.image_url:
                        content_items.append(
                            {"type": "image", "image_url": item.image_url}
                        )

                message_dict["content"] = content_items

            messages.append(message_dict)
        
        if request.dataset:
            project_name = request.dataset.get("project_name")
            version_id = request.dataset.get("version_id")
            bucket_name = request.dataset.get("bucket_name")
            
            if project_name and version_id:
                try:
                    await ensure_dataset_async(project_name, version_id, bucket_name)
                    os.environ["DATA_DIR"] = f"data/{project_name}"
                    os.environ["DATA_VERSION"] = version_id
                    os.environ["BUCKET_NAME"] = bucket_name
                except HTTPException as http_exc:
                    raise http_exc
                except Exception as exc:
                    logger.error(f"Failed to prepare dataset {project_name}@{version_id}: {exc}")
                    raise HTTPException(status_code=500, detail="Dataset preparation failed")

        async def event_generator():
            try:
                async for event in run_agent_workflow(
                    messages,
                    request.debug,
                    request.deep_thinking_mode,
                    request.search_before_planning,
                ):
                    # Check if client is still connected
                    if await req.is_disconnected():
                        logger.info("Client disconnected, stopping workflow")
                        break
                    
                    # Handle human input required event
                    '''if event["event"] == "human_input_required":
                        logger.info(f"human input requirement sent to api")
                        # Store the workflow state
                        workflow_states[event["data"]["workflow_id"]] = {
                            "workflow_id": event["data"]["workflow_id"],
                            "checkpoint_ns": event["data"]["checkpoint_ns"],
                            "state": event["data"]["state"],
                            "timestamp": asyncio.get_event_loop().time()
                        }'''
                    
                    # Always yield the event
                    yield {
                        "event": event["event"],
                        "data": json.dumps(event["data"], ensure_ascii=False),
                    }
            except asyncio.CancelledError:
                logger.info("Stream processing cancelled")
                raise

        return EventSourceResponse(
            event_generator(),
            media_type="text/event-stream",
            sep="\n",
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
