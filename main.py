from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import logging
import json
import time
from dotenv import load_dotenv
from db import DatabaseManager
from transcript_processor import TranscriptProcessor

load_dotenv()

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console_handler)

# FastAPI app setup
app = FastAPI(title="Meeting Summarizer API", description="API for processing and summarizing meeting transcripts", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600
)

db = DatabaseManager()

# ---------------------- Pydantic Models ----------------------
class Transcript(BaseModel):
    id: str
    text: str
    timestamp: str

class MeetingResponse(BaseModel):
    id: str
    title: str

class MeetingDetailsResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    transcripts: List[Transcript]

class MeetingTitleUpdate(BaseModel):
    meeting_id: str
    title: str

class DeleteMeetingRequest(BaseModel):
    meeting_id: str

class SaveTranscriptRequest(BaseModel):
    meeting_title: str
    transcripts: List[Transcript]

class SaveModelConfigRequest(BaseModel):
    provider: str
    model: str
    whisperModel: str
    apiKey: Optional[str] = None

class SaveTranscriptConfigRequest(BaseModel):
    provider: str
    model: str
    apiKey: Optional[str] = None

class TranscriptRequest(BaseModel):
    text: str
    model: str
    model_name: str
    meeting_id: str
    chunk_size: Optional[int] = 5000
    overlap: Optional[int] = 1000
    custom_prompt: Optional[str] = "Generate a summary of the meeting transcript."

# ---------------------- Summary Processor ----------------------
class SummaryProcessor:
    def __init__(self):
        self.db = DatabaseManager()
        self.transcript_processor = TranscriptProcessor()
        logger.info("SummaryProcessor initialized")

    async def process_transcript(self, text: str, model: str, model_name: str, chunk_size: int = 5000, overlap: int = 1000, custom_prompt: str = None):
        if not text.strip():
            raise ValueError("Transcript text is empty")
        if chunk_size <= 0 or overlap < 0:
            raise ValueError("Invalid chunk_size or overlap")
        if overlap >= chunk_size:
            overlap = chunk_size - 1
        num_chunks, all_json_data = await self.transcript_processor.process_transcript(
            text=text,
            model=model,
            model_name=model_name,
            chunk_size=chunk_size,
            overlap=overlap,
            custom_prompt=custom_prompt or "Generate a summary of the meeting transcript."
        )
        return num_chunks, all_json_data

    def cleanup(self):
        try:
            self.transcript_processor.cleanup()
            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)

processor = SummaryProcessor()

# ---------------------- API Endpoints ----------------------
@app.get("/get-meetings", response_model=List[MeetingResponse])
async def get_meetings():
    try:
        meetings = await db.get_all_meetings()
        return [{"id": m["id"], "title": m["title"]} for m in meetings]
    except Exception as e:
        logger.error(f"Error fetching meetings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-meeting/{meeting_id}", response_model=MeetingDetailsResponse)
async def get_meeting(meeting_id: str):
    meeting = await db.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting

@app.post("/save-meeting-title")
async def save_meeting_title(data: MeetingTitleUpdate):
    await db.update_meeting_title(data.meeting_id, data.title)
    return {"message": "Meeting title saved successfully"}

@app.post("/delete-meeting")
async def delete_meeting(data: DeleteMeetingRequest):
    success = await db.delete_meeting(data.meeting_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete meeting")
    return {"message": "Meeting deleted successfully"}

# ---------------------- Transcript Processing ----------------------
async def process_transcript_background(process_id: str, transcript: TranscriptRequest):
    try:
        _, all_json_data = await processor.process_transcript(
            text=transcript.text,
            model=transcript.model,
            model_name=transcript.model_name,
            chunk_size=transcript.chunk_size,
            overlap=transcript.overlap,
            custom_prompt=transcript.custom_prompt
        )
        if all_json_data:
            await db.update_process(process_id, status="completed", result=json.dumps(all_json_data))
        else:
            await db.update_process(process_id, status="failed", error="No chunks processed")
    except Exception as e:
        logger.error(f"Background processing failed: {e}", exc_info=True)
        await db.update_process(process_id, status="failed", error=str(e))

@app.post("/process-transcript")
async def process_transcript_api(transcript: TranscriptRequest, background_tasks: BackgroundTasks):
    process_id = await db.create_process(transcript.meeting_id)
    await db.save_transcript(transcript.meeting_id, transcript.text, transcript.model, transcript.model_name, transcript.chunk_size, transcript.overlap)
    background_tasks.add_task(process_transcript_background, process_id, transcript)
    return {"message": "Processing started", "process_id": process_id}

@app.get("/get-summary/{meeting_id}")
async def get_summary(meeting_id: str):
    result = await db.get_transcript_data(meeting_id)
    if not result:
        raise HTTPException(status_code=404, detail="Meeting ID not found")
    status = result.get("status", "unknown").lower()
    data = json.loads(result["result"]) if result.get("result") else None
    return JSONResponse({
        "status": status,
        "meeting_id": meeting_id,
        "data": data
    })

# ---------------------- Shutdown Event ----------------------
@app.on_event("shutdown")
async def shutdown_event():
    processor.cleanup()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    uvicorn.run("main:app", host="0.0.0.0", port=5167, reload=True)
