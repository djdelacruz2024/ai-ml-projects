import anthropic 
from dotenv import load_dotenv
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready")

def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    text = ""

    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)
        print(f"Tatal pages: {total_pages}")

        for page_num in range(total_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    print(f"Extracted{len(text)} characters")
    return text

def chunk_text(text, chunk_size = 300, overlap = 50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    print(f"Created{len(chunks)} chunks")
    return chunks

def build_index(chunks):
    print("Building search index...")
    embeddings = np.array(embedder.encode(chunks)).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Indexed {index.ntotal} chunks")
    return index

def answer_question(question, chunks, index):
    # Convert question to embedding
    question_embedding = embedder.encode([question]).astype('float32')
    
    # Find most relevant chunks
    distances, indices = index.search(question_embedding, 3)
    
    # Retrieve relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)
    
    # Send to Claude
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Answer the question based on the context below.
If the answer isn't in the context, say you don't have that information.

Context:
{context}

Question: {question}

Answer:"""
            }
        ]
    )
    
    return response.content[0].text

# Get PDF files from the pdfs folder
pdf_folder = "pdfs"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

if not pdf_files:
    print("No PDF files found in pdfs folder")
    exit()

print(f"Found PDFs: {pdf_files}")

# Let user pick a PDF
for i, pdf in enumerate(pdf_files):
    print(f"{i+1}. {pdf}")

choice = int(input("\nSelect a PDF number: ")) - 1
selected_pdf = os.path.join(pdf_folder, pdf_files[choice])

# Extract and chunk the text
text = extract_text_from_pdf(selected_pdf)
chunks = chunk_text(text)

# Build the search index
index = build_index(chunks)

# Question answering loop
print("\nPDF Q&A System Ready")
print("Type 'quit' to exit")
print("-" * 40)

while True:
    question = input("\nYour question: ")
    
    if question.lower() == 'quit':
        print("Goodbye!")
        break
    
    print("\nSearching document...")
    answer = answer_question(question, chunks, index)
    print(f"\nAnswer: {answer}")