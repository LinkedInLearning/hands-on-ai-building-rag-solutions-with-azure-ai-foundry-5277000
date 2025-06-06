import chainlit as cl
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
# Load environment variables
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
azure_openai_embeddings_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
azure_search_service_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
azure_search_service_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
embedding_endpoint =  os.getenv("EMBEDDING_ENDPOINT_URL")

# Initialize Azure OpenAI client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version=azure_openai_api_version,
)

def run_rag(chat_prompt):
    """
    Executes a Retrieval-Augmented Generation (RAG) process using Azure OpenAI and Azure Cognitive Search.

    Args:
        chat_prompt (list): A list of message dictionaries representing the chat history, formatted for the OpenAI API.

    Returns:
        str: The generated response from the Azure OpenAI model, augmented with retrieved documents from Azure Search.

    This function sends the chat prompt to the Azure OpenAI chat completion endpoint, using Azure Cognitive Search as a data source
    for retrieval-augmented generation. It configures the search parameters, including semantic configuration, vector search, and
    authentication, and returns the model's response.
    """
    messages = chat_prompt

    # Generate the completion
    completion = client.chat.completions.create(
        model=azure_openai_deployment,
        messages=messages,
        max_tokens=800,
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        extra_body={
        "data_sources": [{
            "type": "azure_search",
            "parameters": {
                "endpoint": azure_search_service_endpoint,
                "index_name": azure_search_service_index_name,
                "semantic_configuration": "hikingproductsrag-semantic-configuration",
                "query_type": "vector_semantic_hybrid",
                "fields_mapping": {
                "content_fields_separator": "\n",
                "content_fields": [
                    "chunk"
                ],
                "filepath_field": "title",
                "title_field": "chunk_id",
                "url_field": None,
                "vector_fields": [
                    "text_vector"
                ]
                },
                "in_scope": True,
                "filter": None,
                "strictness": 3,
                "top_n_documents": 5,
                "authentication": {
                "type": "api_key",
                "key": azure_search_service_admin_key
                },
                "embedding_dependency": {
                "type": "endpoint",
                "endpoint": embedding_endpoint,
                "authentication": {
                    "type": "api_key",
                    "key": azure_openai_key
                }
                }
            }
            }]
        }
    )

    return completion.choices[0].message.content

# Function to add a conversation turn to the chat_prompt
def add_to_chat_prompt(chat_prompt, role, content):
    """
    Adds a new message to the chat_prompt list.

    Args:
        chat_prompt: List of message dicts (existing chat history).
        role: 'user' or 'assistant'.
        content: The message content.

    Returns:
        The updated chat_prompt list (with the new message appended).
    """
    chat_prompt.append({"role": role, "content": content})
    return chat_prompt

@cl.on_chat_start
def on_chat_start():
    # Initialize the chat prompt with a system message
    chat_prompt = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information."
        }
    ]
    # Store the chat prompt in the user session
    cl.user_session.set("chat_prompt", chat_prompt)

@cl.on_message
async def main(message: cl.Message):
    
    # Get the system message
    chat_prompt = cl.user_session.get("chat_prompt")
        
    # Add the user's message to the chat prompt
    chat_prompt = add_to_chat_prompt(chat_prompt, "user", message.content)
    
    # Run the RAG process
    response = run_rag(chat_prompt)
    
    # Send a response back to the user
    await cl.Message(
        content=response,
    ).send()

    # Add the assistant's response to the chat prompt
    chat_prompt = add_to_chat_prompt(chat_prompt, "assistant", response)

    