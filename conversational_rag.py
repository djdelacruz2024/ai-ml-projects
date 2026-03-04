import anthropic
from dotenv import load_dotenv
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready")

# Healthcare documents
documents = [
    "Patient John Smith, age 45, diagnosed with Type 2 Diabetes in 2019. Currently on Metformin 500mg twice daily. Last HbA1c reading was 7.2% recorded in March 2024.",
    "Patient Sarah Johnson, age 32, presents with hypertension. Prescribed Lisinopril 10mg once daily. Blood pressure consistently around 130/85 over last 6 months.",
    "Patient Michael Chen, age 58, history of coronary artery disease. Had stent placement in 2021. Currently on Aspirin 81mg and Atorvastatin 40mg daily.",
    "Hospital policy states that all patient data must be encrypted at rest and in transit. Access to patient records requires two factor authentication.",
    "Insurance claim process requires diagnosis code, procedure code, and attending physician signature. Claims must be submitted within 90 days of service.",
    "Patient Emily Davis, age 28, diagnosed with anxiety disorder. Prescribed Sertraline 50mg daily. Follow up appointment scheduled for next month.",
    "Elevance Health covers preventive care at 100% for in network providers. Annual wellness visits, vaccinations, and cancer screenings are fully covered.",
    "Prior authorization is required for MRI and CT scan procedures. Requests must include clinical notes justifying medical necessity.",
]

print("Indexing documents...")
document_embeddings = np.array(embedder.encode(documents)).astype('float32')
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)
print(f"Indexed {index.ntotal} documents")
conversation_history = []

def retrieve_context(question, top_k=2):
    # Convert question to embedding
    question_embedding = embedder.encode([question]).astype('float32')
    
    # Search for most relevant documents
    distances, indices = index.search(question_embedding, top_k)
    
    # Return relevant documents
    relevant_docs = [documents[i] for i in indices[0]]
    return "\n\n".join(relevant_docs)

def chat_with_rag(user_message):
    # Retrieve relevant context for this message
    context = retrieve_context(user_message)
    
    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    # Build system prompt with retrieved context
    system_prompt = f"""You are a helpful healthcare AI assistant for Elevance Health.
Answer questions based on the context provided below.
If the answer isn't in the context, say you don't have that information.
Always be accurate and professional when discussing patient or medical information.

Relevant Context:
{context}"""
    
    # Send full conversation history to Claude
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=conversation_history
    )
    
    assistant_message = response.content[0].text
    
    # Add response to history
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message

print("\nElevance Health AI Assistant")
print("Ask me about patients, coverage, or claims")
print("Type 'quit' to exit")
print("-" * 40)

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    response = chat_with_rag(user_input)
    print(f"\nAssistant: {response}")