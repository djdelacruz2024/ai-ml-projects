import anthropic
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

conversation_history = []

def chat(user_message):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system="You are a helpful AI assistant. You remember everything said earlier in the conversation and refer back to it when relevant.",
        messages=conversation_history
    )
    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    return assistant_message

print("AI Chatbot with Memory")
print("Type 'quit' to exit")
print("Type 'history' to see conversation history")
print("-" * 40)

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    if user_input.lower() == 'history':
        print("\n--- CONVERSATION HISTORY ---")
        for message in conversation_history:
            role = "You" if message["role"] == "user" else "Claude"
            print(f"{role}: {message['content']}")
        continue
    
    response = chat(user_input)
    print(f"\nClaude: {response}")