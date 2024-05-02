import logging
from datetime import datetime, timedelta
from typing import List
import pytz

import logging
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build
import json
import logging
import os
import pickle
from google.auth.transport.requests import Request as RequestGoogle
from google_auth_oauthlib.flow import InstalledAppFlow
import requests
from langchain_core.tools import tool


SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]

# File to store the user's access and refresh tokens
TOKEN_PICKLE_FILE = "token.pickle"


def load_credentials_scheduling():
    try:
        creds = None
        # Load the credentials from the saved token
        if os.path.exists(TOKEN_PICKLE_FILE):
            with open(TOKEN_PICKLE_FILE, "rb") as token:
                creds = pickle.load(token)
        # If there are no valid credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(RequestGoogle())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "your_client_secret.json",
                    SCOPES,
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(TOKEN_PICKLE_FILE, "wb") as token:
                pickle.dump(creds, token)
        return creds
    except Exception as e:
        logging.error(f"An error occurred in load_credentials: {e}")
        raise e


def load_credentials():
    # Path to your service account key file
    SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # Define the scopes
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    # Authenticate and construct service
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        return credentials
    except Exception as e:
        print(f"Failed to authenticate with Google Cloud: {e}")
        raise e


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def schedule_meeting(
    start_time: str, subject: str = None, description: str = "", attendees: List = []
):
    """
    Schedule a meeting on the user's calendar based on the start and end time.
    """
    try:
        meeting_summary = subject
        creds = load_credentials_scheduling()
        service = build("calendar", "v3", credentials=creds)

        timezone = "America/New_York"
        tz = pytz.timezone(timezone)
        start_time = datetime.fromisoformat(start_time).astimezone(tz)
        end_time = start_time + timedelta(hours=1)
        event = {
            "summary": meeting_summary,
            "description": description,
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": timezone,
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": timezone,
            },
            "attendees": [{"email": attendee} for attendee in attendees],
            "conferenceData": {
                "createRequest": {
                    "requestId": f"{start_time.timestamp()}",  # Unique ID for the request
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            },
        }

        # Use the 'conferenceDataVersion' query parameter to ensure response includes conferenceData
        event = (
            service.events()
            .insert(calendarId="primary", body=event, conferenceDataVersion=1)
            .execute()
        )

        logging.info(f"Event created: {event.get('htmlLink')}")
        return event
    except Exception as e:
        logging.error(f"Failed to schedule meeting: {e}")
        return None
