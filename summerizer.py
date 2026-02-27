import anthropic
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# Initialize the client
def summarize(text, style):
    if style == "bullets":
        prompt = f"Summarize the following text in 3 concise bullet points:\n\n{text}"
    elif style == "short":
        prompt = f"Summarize the following text in one sentence:\n\n{text}"
    elif style == "detailed":
        prompt = f"Provide a detailed summary of the following text in 2-3 paragraphs:\n\n{text}"
    elif style == "simple":
        prompt = f"Summarize the following text in simple language a teenager that is sarcastic would understand:\n\n{text}"

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

# Test all four styles
text = """
Artificial intelligence has rapidly transformed numerous industries over the past decade. 
In healthcare, AI systems can now detect diseases from medical images with accuracy 
rivaling experienced physicians. In finance, machine learning algorithms process millions 
of transactions per second to detect fraudulent activity. The transportation sector is 
being revolutionized by self-driving vehicles that use computer vision and deep learning 
to navigate complex environments. Education is being personalized through adaptive 
learning systems that adjust to individual student needs. However, these advances come 
with significant challenges. Questions about job displacement, algorithmic bias, data 
privacy, and the concentration of AI power among a few large corporations have sparked 
intense debate among policymakers, ethicists, and technologists. The next decade will 
likely determine how humanity chooses to govern and integrate these powerful technologies 
into the fabric of society.
"""

styles = ["bullets", "short", "detailed", "simple"]

for style in styles:
    print(f"\n--- {style.upper()} SUMMARY ---")
    print(summarize(text, style))
