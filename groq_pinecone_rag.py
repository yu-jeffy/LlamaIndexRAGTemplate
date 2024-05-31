# Linux Sqlite >3.35 fix
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI
import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.response.pprint_utils import pprint_source_node
from pinecone import Pinecone, ServerlessSpec, PineconeException
import asyncio
import time

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")

client = OpenAI()

api_key = os.getenv("GROQ_API_KEY")

llm = Groq(model="llama3-70b-8192", api_key=api_key)

indexes = pc.list_indexes()
print(indexes)

pinecone_index = pc.Index("testindex")

async def delete_index():
    try:
        pinecone_index.delete(deleteAll="true")
    except PineconeException as e:
        print(f"Failed to delete index: {e}")

# Create a new event loop
loop = asyncio.new_event_loop()

# Set the new event loop as the current one
asyncio.set_event_loop(loop)

try:
    # Run the delete operation in the event loop
    loop.run_until_complete(delete_index())
finally:
    # Close the loop after we're done
    loop.close()

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# load documents
filename_fn = lambda filename: {"file_name": filename}

# automatically sets the metadata of each document according to filename_fn
documents = SimpleDirectoryReader(
    "./test/", file_metadata=filename_fn
).load_data()


vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, text_key="content"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, text_key="content"
)
retriever = VectorStoreIndex.from_vector_store(vector_store).as_retriever(
    similarity_top_k=4,
    verbose=True
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(llm=llm, response_mode="refine")

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
# response = query_engine.query("What happens in chapter 3?")
# print(response)

SYSTEM_PROMPT = """
You are assisting in authoring a novel. You are tasked with brainstorming and helping to write the book.
"""

# Initialize the conversation with a system message
messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]

while True:
    # Get user input
    user_input = input("User: ")

    # Add user message to the conversation history
    messages.append(ChatMessage(role="user", content=user_input))

    # Convert user input into a vector using the same model used for the embeddings
    user_vector = embed_model.get_text_embedding([user_input])

    # Query the Pinecone index using the user vector
    resp = pinecone_index.query(user_vector, top_k=1)

    # Get the document ID of the top result
    top_result_id = resp.ids[0][0]

    # Get the document corresponding to the top result
    top_result_document = documents[top_result_id]

    # Add the document content to the conversation history
    messages.append(ChatMessage(role="assistant", content=top_result_document))

    # Use the document content as context for the next query to the language model
    context = top_result_document

    # Query the language model with the user's input and the context
    llm_response = llm.query(prompt=user_input, context=context)

    # Add the language model's response to the conversation history
    messages.append(ChatMessage(role="assistant", content=llm_response))

    # Print the language model's response
    print("llm: ", llm_response)