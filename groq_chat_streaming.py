from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os
import sys

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = Groq(model="llama3-70b-8192", api_key=api_key)

# response = llm.complete("Explain the importance of low latency LLMs")

# print(response)

def chat_with_llm():
    print("Starting the chat with LLM. Type 'exit' to end the chat.\n")
    
    # Initialize with a system message using ChatMessage objects
    messages = [ChatMessage(role="system", content="You are a helpful assistant.")]

    while True:
        print()
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Exiting chat...")
            break

        # Append user's message as a ChatMessage object
        messages.append(ChatMessage(role="user", content=user_input))

        try:
            # Use stream_chat API with ChatMessage objects directly
            responses = llm.stream_chat(messages)
            print("LLM:", end=" ")
            for response in responses:
                print(response.delta, end="")
        except Exception as e:
            print(f"An error occurred: {e}")
            continue  # Continue despite errors to allow further attempts
        
        # Optionally clear messages to keep only the last message or reset for full history
        # messages = [messages[-1]]  # Keeps the last message

if __name__ == "__main__":
    chat_with_llm()