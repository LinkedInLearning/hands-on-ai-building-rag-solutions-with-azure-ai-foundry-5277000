
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

user_input = "Do you have other similar products?"

#Prepare the chat prompt
chat_prompt = [
  {
    "role": "system",
    "content": "You are an AI assistant that helps people find information."
  },
  {
    "role": "user",
    "content": "What is the price of the UrbanTraveler?"
  },
  {
    "role": "assistant",
    "content": "The price of the UrbanTraveler Laptop Backpack is $90 [doc1]."
  },
  {
    "role": "user",
    "content": "What are its features?"
  },
  {
    "role": "assistant",
    "content": "The UrbanTraveler Laptop Backpack includes the following features:\n\n- **Durable Material**: Constructed from high-quality polyester for durability and water resistance [doc1].\n- **Laptop Compartment**: Padded compartment fits laptops up to 17 inches [doc1][doc2].\n- **Multiple Storage Options**: Main compartment, front pocket, side pockets, and an organizer panel for small items [doc1].\n- **USB Charging Port**: Built-in USB port for convenient on-the-go device charging [doc1][doc3].\n- **Anti-Theft Design**: Hidden zippers and an RFID-blocking pocket enhance security [doc1].\n- **Comfortable Fit**: Adjustable shoulder straps and a chest strap ensure a secure and comfortable fit [doc1][doc2].\n- **Ventilated Back Panel**: Mesh back panel promotes airflow for added comfort [doc1].\n- **Luggage Strap**: Back strap allows secure attachment to rolling luggage handles [doc1].\n\nThese features make the backpack ideal for travel and daily commutes, offering convenience, organization, and security."
  },
  {
    "role": "user",
    "content": user_input
  }
]

# Include speech result if speech is enabled
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

print(completion.to_json())