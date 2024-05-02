from fastapi import FastAPI
from pydantic import BaseModel
from scheduler import schedule_meeting
from llm import extract_meeting_time

app = FastAPI()


class QueryInput(BaseModel):
    input: str


@app.post("/schedule")
def scheduler(body: QueryInput):
    result = extract_meeting_time(body.input)
    time_selected = result["time_selected"]
    attendees = result["attendees"]
    subject = result["subject"]
    meeting_duration = result["meeting_duration"]
    print(result)

    schedule_meeting(
        start_time=result["time_selected"],
        attendees=result["attendees"],
        subject=result["subject"],
    )
    return {"answer": "Meeting scheduled successfully"}
