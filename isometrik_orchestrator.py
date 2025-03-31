import asyncio
import os
import requests
import json
from dotenv import load_dotenv
from typing import List, Dict, Optional
from multi_agent_orchestrator.orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorConfig,
)
from multi_agent_orchestrator.classifiers import (
    OpenAIClassifier,
    OpenAIClassifierOptions,
)
from multi_agent_orchestrator.storage import InMemoryChatStorage
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from multi_agent_orchestrator.agents import (
    AgentResponse,
    Agent,
    AgentOptions,
)


load_dotenv()

memory_storage = InMemoryChatStorage()


class IsometrikAgentOptions(AgentOptions):
    def __init__(
        self,
        name: str,
        description: str,
        api_url: str,
        auth_token: str,
        additional_params: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name=name, description=description)
        self.api_url = api_url
        self.auth_token = auth_token
        self.additional_params = additional_params
class IsometrikAgent(Agent):
    def __init__(self, options: IsometrikAgentOptions):
        super().__init__(options)
        self.api_url = options.api_url
        self.auth_token = options.auth_token
        self.additional_params = options.additional_params
    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None,
    ) -> AgentResponse:
        """Process a request by sending it to the Isometrik API."""
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "session_id": session_id,
            "message": input_text,
            "agent_id": self.additional_params.agent_id,
            "isLoggedIn": self.additional_params.isLoggedIn,
            "finger_print_id": self.additional_params.finger_print_id,
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # print(data)
                return ConversationMessage(
                    role=ParticipantRole.ASSISTANT.value,
                    content=[data]
                )
        
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            return AgentResponse(
                metadata={},
                streaming=False,
                output=error_msg
            )

def create_query_agent(body):
    options = IsometrikAgentOptions(
        name="Query Agent",
        description="Specializes in answering customer FAQs and product information",
        api_url=os.getenv("QUERY_AGENT_API_URL"),
        auth_token=os.getenv("QUERY_AGENT_AUTH_TOKEN"),
        additional_params=body
    )
    return IsometrikAgent(options)



def create_order_agent(body):
    options = IsometrikAgentOptions(
        name="Order Agent",
        description="Specializes in handling order-related queries, order status, and processing new orders",
        api_url=os.getenv("ORDER_AGENT_API_URL"),
        auth_token=os.getenv("ORDER_AGENT_AUTH_TOKEN"),
        additional_params=body
    )
    return IsometrikAgent(options)


def create_ecom_manager_agent(body):
    options = IsometrikAgentOptions(
        name="Ecom Manager Agent",
        description="Specializes in finding toxin-free, eco-friendly solutions for home and personal care",
        api_url=os.getenv("MANAGER_AGENT_API_URL"),
        auth_token=os.getenv("MANAGER_AGENT_AUTH_TOKEN"),
        additional_params=body
    )
    return IsometrikAgent(options)


def create_subscription_agent(body):
    options = IsometrikAgentOptions(
        name="Subscription Agent",
        description="Specializes in handling subscription-related queries",
        api_url=os.getenv("SUBSCRIPTION_AGENT_API_URL"),
        auth_token=os.getenv("SUBSCRIPTION_AGENT_AUTH_TOKEN"),
        additional_params=body
    )
    return IsometrikAgent(options)


def create_product_details_agent(body):
    options = IsometrikAgentOptions(
        name="Product Details Agent",
        description="Specializes in providing detailed product information, recommendations, comparisons, and purchase advice.",
        api_url=os.getenv("PRODUCT_AGENT_API_URL"),
        auth_token=os.getenv("PRODUCT_AGENT_AUTH_TOKEN"),
        additional_params=body
    )
    return IsometrikAgent(options)



def create_orchestrator(body):
    """Creates and initializes the orchestrator with all agents with streaming support."""
    custom_openai_classifier = OpenAIClassifier(
        OpenAIClassifierOptions(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_id="gpt-4o-mini",
        )
    )

    orchestrator = MultiAgentOrchestrator(
        options=OrchestratorConfig(
            LOG_CLASSIFIER_OUTPUT=True,
            LOG_CLASSIFIER_RAW_OUTPUT=True,
            MAX_RETRIES=2,
            USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
            MAX_MESSAGE_PAIRS_PER_AGENT=8,
        ),
        classifier=custom_openai_classifier,
        # storage=memory_storage,
    )

    query_agent = create_query_agent(body)
    order_agent = create_order_agent(body)
    manager_agent = create_ecom_manager_agent(body)
    subscription_agent = create_subscription_agent(body)
    product_agent = create_product_details_agent(body)

    orchestrator.add_agent(query_agent)
    orchestrator.add_agent(order_agent)
    orchestrator.add_agent(manager_agent)
    orchestrator.add_agent(subscription_agent)
    orchestrator.add_agent(product_agent)

    orchestrator.classifier.set_system_prompt(
        """
        You are AgentMatcher, an intelligent assistant designed to analyze user queries and match them with
        the most suitable agent or department. Your task is to understand the user's request,
        identify key entities and intents, and determine which agent would be best equipped
        to handle the query.

        Important: The user's input may be a follow-up response to a previous interaction.
        The conversation history, including the name of the previously selected agent, is provided.
        If the user's input appears to be a continuation of the previous conversation
        (e.g., "yes", "ok", "I want to know more", "1"), select the same agent as before.

        Analyze the user's input and categorize it into one of the following agent types:
        <agents>
        {{AGENT_DESCRIPTIONS}}
        </agents>

        Guidelines for classification:

        1. Agent Type: Choose the most appropriate agent based on the nature of the query.
           - Query Agent: General product information including greetings, FAQs, and basic inquiries
           - Order Agent: Order status, tracking, processing, and shipping
           - Eco Manager Agent: Eco-friendly products, toxin-free solutions, sustainable living
           - Subscription Agent: Subscription-related queries
           - Product Details Agent: Detailed product information, recommendations, comparisons, and purchase advice.

        2. Priority: Assign based on urgency and impact.
           - High: Issues affecting service, urgent requests, or critical product information.
           - Medium: Non-urgent product inquiries, general questions, or subscription management.
           - Low: Information requests, browsing, or vague inquiries.

        3. Key Entities: Extract important product names, issues, or specific requests mentioned.
           For follow-ups, include relevant entities from previous interactions.

        4. Confidence: Indicate how confident you are in the classification.
           - High: Clear, straightforward requests or clear follow-ups.
           - Medium: Requests with some ambiguity but likely classification.
           - Low: Vague or multi-faceted requests that could fit multiple categories.

        5. Is Followup: Indicate whether the input is a follow-up to a previous interaction.

        Specific Classification Rules:
        - If the user asks about general information or product searches or perticular product or category → Query Agent.
        - If the user asks about order status, tracking, or shipping, past orders → Order Agent.
        - If the user asks about subscription management, billing, or account status → Subscription Agent.
        - If the user asks for detailed product specifications, recommendations, comparisons, or purchase advice , main target is product or category recommendations → Product Details Agent.
        - If the user asks about eco-friendly alternatives, sustainable products, or environmental impact → Eco Manager Agent.

        For short responses like "yes", "ok", "I want to know more", or numerical answers,
        treat them as follow-ups and maintain the previous agent selection.

        Here is the conversation history that you need to take into account before answering:
        <history>
        {{HISTORY}}
        </history>

        Skip any preamble and provide only the response in the specified format.
        """
    )

    return orchestrator
