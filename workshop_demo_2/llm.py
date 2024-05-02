import os

from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel, Field

from google.oauth2 import service_account
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Tool,
)

load_dotenv()
# Path to your service account key file
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIAL")

# Define the scopes
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Authenticate and construct service
try:
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    print(credentials)
except Exception as e:
    print(f"Failed to authenticate with Google Cloud: {e}")
    raise e


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


def llm_call(system_prompt, input_dict, run_name, tools=None, params=None):
    """
    Calls the large language model with the user input query and returns the raw result.

    :param input_query: The user's query as a string.
    :return: Raw result from the large language model.
    """
    model = GenerativeModel("gemini-1.5-pro-preview-0409")

    schedule_meeting_tool = Tool(
        function_declarations=[tools],
    )

    response = model.generate_content(
        input_dict,
        tools=[schedule_meeting_tool],
    )
    # if params is None:
    #     params = {
    #         "temperature": 1.3,
    #         "max_tokens": 1000,
    #     }

    # llm = ChatVertexAI(
    #     credentials=credentials,
    #     model_name="gemini-1.5-pro-preview-0409",
    #     project="prj-uc-genai-labs",
    #     region="us-central1",
    # )
    # tools = [GetWeather]
    # prompt = ChatPromptTemplate.from_template(system_prompt)
    # llm_with_tools = llm.bind_tools(tools)

    # configured_llm = llm_with_tools.with_config({"run_name": "Extract Meeting Time"})
    # chain = prompt | configured_llm
    # response = chain.invoke(input_dict)

    response = response.candidates[0].content.parts[0]
    response = response.to_dict()
    print(type(response))
    print(response)
    if tools:
        response = parse_results(response)
    return response


def extract_meeting_time(query):
    """
    Calls the large language model with the user input query and returns the selected.

    :param input_query: The user's query as a string.
    :return: Raw result from the large language model.
    """
    system_prompt = """
    Analyze the user's email and output parameters needed to schedule a meeting.

    Output json.

    Message history:
    {message_history}
    """

    tools = FunctionDeclaration(
        name="extract_meeting_parameters",
        description="""Extracts the meeting parameters from the users query.
                                """,
        parameters={
            "type": "object",
            "properties": {
                "time_selected": {
                    "type": "string",
                    "description": "The meeting time selected by the user. Must be a string in ISO 8601 format. If there is no meeting time, return None.",
                },
                "meeting_duration": {
                    "type": "string",
                    "description": "The duration of the meeting. Must be a string in ISO 8601 format.",
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The email addresses of the attendees.",
                },
                "subject": {
                    "type": "string",
                    "description": "The subject of the meeting.",
                },
            },
            "required": [
                "time_selected",
                "meeting_duration",
                "attendees",
                "subject",
            ],
        },
    )

    # tools = [ScheduleMeetingParameters]

    input_dict = {
        "input": query,
        "message_history": [],
    }
    response = llm_call(
        system_prompt=system_prompt,
        input_dict=query,
        run_name="Extract Meeting Time",
        tools=tools,
    )
    scheduled_time = response
    print(f"Extracted meeting time: {scheduled_time}")
    return scheduled_time


def parse_results(llm_result: dict) -> dict:

    if (
        "function_call" in llm_result
        and llm_result["function_call"]["name"] == "extract_meeting_parameters"
    ):
        args = llm_result["function_call"]["args"]
        time_selected = args.get("time_selected", None)
        attendees = args.get("attendees", [])
        subject = args.get("subject", "")
        meeting_duration = args.get("meeting_duration", "")
        return {
            "time_selected": time_selected,
            "attendees": attendees,
            "subject": subject,
            "meeting_duration": meeting_duration,
        }
    else:
        return {}
