import anthropic
from dotenv import load_dotenv
import os
from flask import Flask, render_template, request, jsonify

load_dotenv()

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

conversation_history = []

def find_relevant_docs(question, top_k=2):
    # Simple keyword matching as lightweight alternative to embeddings
    question_words = set(question.lower().split())
    scored_docs = []
    
    for doc in documents:
        doc_words = set(doc.lower().split())
        score = len(question_words.intersection(doc_words))
        scored_docs.append((score, doc))
    
    scored_docs.sort(reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

def chat_with_rag(user_message):
    relevant_docs = find_relevant_docs(user_message)
    context = "\n\n".join(relevant_docs)
    
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    system_prompt = f"""You are a helpful healthcare AI assistant.
Answer questions based on the context provided below.
If the answer isn't in the context, say you don't have that information.

Relevant Context:
{context}"""
    
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=conversation_history
    )
    
    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data['message']
    response = chat_with_rag(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)