import anthropic
from dotenv import load_dotenv
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()  # Load environment variables from .env file
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully")

# Our "database" of healthcare documents
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

print(f"Loaded {len(documents)} documents")

print("Creating embeddings for all documents...")
document_embeddings = embedder.encode(documents)
document_embeddings = np.array(document_embeddings).astype('float32')
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)
print(f"Created embeddings with dimesion: {dimension}")
print(f"Total documents indexed: {index.ntotal}")


def retrieve_and_answer(question, top_k=2):
    # Convert the question to an embedding
    question_embedding = embedder.encode([question]).astype('float32')
    
    # Search for the most similar documents
    distances, indices = index.search(question_embedding, top_k)
    
    # Retrieve the actual document text
    relevant_docs = [documents[i] for i in indices[0]]
    
    # Build context from retrieved documents
    context = "\n\n".join(relevant_docs)
    
    # Send to Claude with the retrieved context
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Answer the question based only on the provided context.
                
Context:
{context}

Question: {question}

Answer:"""
            }
        ]
    )
    
    return response.content[0].text, relevant_docs

# Test it with healthcare questions
questions = [
    "What medication is John Smith taking for diabetes?",
    "Does Elevance Health cover annual wellness visits?",
    "What is required to submit an insurance claim?",
    "What is Sarah Johnson's blood pressure medication?",
]

for question in questions:
    print(f"\nQuestion: {question}")
    answer, retrieved_docs = retrieve_and_answer(question)
    print(f"Answer: {answer}")
    print(f"Retrieved from: {retrieved_docs[0][:80]}...")
    print("-" * 60)