import uuid
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import json
from isometrik_orchestrator import create_orchestrator
from multi_agent_orchestrator.types import ConversationMessage
from multi_agent_orchestrator.agents import AgentResponse

app = FastAPI(title="Isometrik API", description="Isometrik API for trulyfree bot with multi-agent orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    content: str
    user_id: str
    session_id: str


def setup_orchestrator():
    orchestrator = create_orchestrator()

    return orchestrator


async def start_generation(query, user_id, session_id):
    try:
        orchestrator = setup_orchestrator()
        response = await orchestrator.route_request(query, user_id, session_id)

        if isinstance(response, AgentResponse):
            if isinstance(response.output, ConversationMessage):
                response_text = response.output.content[0].get("text", "")
                
                if '"productId"' in response_text:
                    try:
                        extracted_products = json.loads(response_text)
                        return {
                            "response": "We've found exactly what you're looking for! Check out your search results below.",
                            "request_id": str(uuid.uuid4()),
                            "widgets": [
                                {
                                    "widgetId": 2,
                                    "widgets_type": 2,
                                    "type": "products",
                                    "widget": extracted_products
                                },
                                {
                                    "widgetId": 1,
                                    "widgets_type": 1,
                                    "type": "options",
                                    "widget": ["Load More Products"]
                                }
                            ]
                        }
                    except (json.JSONDecodeError, re.error) as e:
                        print(f"Error extracting products: {e}")
                        pass
                
                if '"masterOrderId"' in response_text:
                    try:
                        extracted_orders = json.loads(response_text)
                        options = [
                            "Load More Orders",
                            "Last Month",
                            "Last 3 Months",
                            "Last 6 Months"
                        ]
                        return {
                                "response": "Here are your recent orders:",
                                "request_id": str(uuid.uuid4()),
                                "widgets": [
                                    {
                                        "widgetId": 3,
                                        "widgets_type": 3,
                                        "type": "orders",
                                        "widget": extracted_orders
                                    },
                                    {
                                        "widgetId": 1,
                                        "widgets_type": 1,
                                        "type": "options",
                                        "widget": options
                                    }
                                ]
                            }
                    except (json.JSONDecodeError, re.error) as e:
                        print(f"Error extracting orders: {e}")
                        pass
                
                if '"subscriptionId"' in response_text and '"storeId"' in response_text:
                    try:
                        extracted_subscriptions = json.loads(response_text)
                        options = [
                            "View All Subscriptions",
                            "Active Subscriptions",
                            "Paused Subscriptions",
                            "Cancelled Subscriptions"
                        ]
                        return {
                                "response": "Here are your subscriptions:",
                                "request_id": str(uuid.uuid4()),
                                "widgets": [
                                    {
                                        "widgetId": 4,
                                        "widgets_type": 4,
                                        "type": "subscriptions",
                                        "widget": extracted_subscriptions
                                    },
                                    {
                                        "widgetId": 1,
                                        "widgets_type": 1,
                                        "type": "options",
                                        "widget": options
                                    }
                                ]
                            }
                    except (json.JSONDecodeError, re.error) as e:
                        print(f"Error extracting subscriptions: {e}")
                        pass
                
                if isinstance(response_text, str) and '"text"' in response_text and '"options"' in response_text:
                    text_match = re.search(r'"text":\s*"([^"]*)"', response_text)
                    extracted_text = text_match.group(1) if text_match else ""
                    
                    options_match = re.search(r'"options":\s*\[(.*?)\]', response_text)
                    if options_match:
                        options_str = options_match.group(1)
                        extracted_options = [opt.strip('"') for opt in re.findall(r'"([^"]*)"', options_str)]
                    else:
                        extracted_options = []      
                    
                    return {
                        "response": extracted_text,
                        "request_id": str(uuid.uuid4()),
                        "widgets": [
                            {
                                "widgetId": 1,
                                "widgets_type": 1,
                                "type": "options",
                                "widget": extracted_options
                            }
                        ]
                    }
                
                return response.output
            return response.output
        return response
    except Exception as e:
        print(f"Error in start_generation: {e}")
        return {"error": str(e)}




@app.post("/agent-chat")
async def stream_chat(body: ChatRequest):
    response = await start_generation(body.content, body.user_id, body.session_id)
    return response


@app.get("/")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
