import anthropic
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import json

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

np.random.seed(42)
n_claims = 500

data = {
    'claim_id': range(1, n_claims + 1),
    'patient_age': np.random.randint(18, 85, n_claims),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_claims),
    'diagnosis_code': np.random.choice(['E11', 'I10', 'J45', 'M54', 'F32'], n_claims),
    'procedure': np.random.choice(['office_visit', 'mri', 'blood_test', 'xray', 'specialist'], n_claims),
    'claim_amount': np.random.uniform(100, 5000, n_claims).round(2),
    'approved': np.random.choice([True, False], n_claims, p=[0.85, 0.15]),
    'processing_days': np.random.randint(1, 30, n_claims)
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/insurance_claims.csv', index=False)
print("Dataset created successfully")
print(df.head())
print(f"\nShape: {df.shape}")

def analyze_data(question):
    # Convert dataframe info to string for context
    data_context = f"""
You have access to a pandas DataFrame called 'df' with insurance claims data.
Shape: {df.shape}
Columns: {list(df.columns)}
Data types: {df.dtypes.to_dict()}
Sample data:
{df.head().to_string()}

Write Python code using pandas to answer this question: {question}

Rules:
- Use only pandas and basic Python
- Store your final answer in a variable called 'result'
- Make result a simple string, number, or dictionary
- Do not use print statements
- Do not import anything
"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": data_context
            }
        ]
    )
    
    # Extract the code from response
    code = response.content[0].text
    
    # Clean up code blocks if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    
    # Execute the generated code
    local_vars = {"df": df, "pd": pd, "np": np}
    exec(code, local_vars)
    result = local_vars.get("result", "No result found")
    
    return result, code

# Test questions
questions = [
    "What is the average claim amount by region?",
    "What is the most common diagnosis code?",
    "What percentage of claims were approved?",
    "What is the average processing days for approved vs denied claims?",
]

print("\nAI Data Analyst")
print("-" * 40)

for question in questions:
    print(f"\nQuestion: {question}")
    result, code = analyze_data(question)
    print(f"Answer: {result}")


# Interactive mode
print("\nInteractive Data Analysis Mode")
print("Ask any question about the insurance claims data")
print("Type 'quit' to exit")
print("-" * 40)

while True:
    question = input("\nYour question: ")
    
    if question.lower() == 'quit':
        print("Goodbye!")
        break
    
    print("Analyzing...")
    result, code = analyze_data(question)
    print(f"Answer: {result}")