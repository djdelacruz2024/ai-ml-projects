import anthropic
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("AI Agent initialized")

# Tool functions the agent can call
def calculate_bmi(weight_kg, height_m):
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return {"bmi": round(bmi, 2), "category": category}

def get_patient_info(patient_name):
    patients = {
        "john smith": {
            "age": 45,
            "condition": "Type 2 Diabetes",
            "medication": "Metformin 500mg",
            "last_visit": "March 2024"
        },
        "sarah johnson": {
            "age": 32,
            "condition": "Hypertension",
            "medication": "Lisinopril 10mg",
            "last_visit": "January 2024"
        },
        "michael chen": {
            "age": 58,
            "condition": "Coronary Artery Disease",
            "medication": "Aspirin 81mg",
            "last_visit": "February 2024"
        }
    }
    patient = patients.get(patient_name.lower())
    if patient:
        return patient
    return {"error": f"Patient {patient_name} not found"}

def check_coverage(procedure):
    coverage = {
        "annual wellness visit": {"covered": True, "coverage_percent": 100},
        "mri": {"covered": True, "coverage_percent": 80, "requires_auth": True},
        "ct scan": {"covered": True, "coverage_percent": 80, "requires_auth": True},
        "vaccination": {"covered": True, "coverage_percent": 100},
        "cancer screening": {"covered": True, "coverage_percent": 100},
        "specialist visit": {"covered": True, "coverage_percent": 70},
    }
    result = coverage.get(procedure.lower())
    if result:
        return result
    return {"covered": False, "message": "Procedure not found in coverage database"}

tools = [
    {
        "name": "calculate_bmi",
        "description": "Calculate BMI and weight category given weight in kg and height in meters",
        "input_schema": {
            "type": "object",
            "properties": {
                "weight_kg": {
                    "type": "number",
                    "description": "Weight in kilograms"
                },
                "height_m": {
                    "type": "number",
                    "description": "Height in meters"
                }
            },
            "required": ["weight_kg", "height_m"]
        }
    },
    {
        "name": "get_patient_info",
        "description": "Get patient information including age, condition, and medication",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_name": {
                    "type": "string",
                    "description": "Full name of the patient"
                }
            },
            "required": ["patient_name"]
        }
    },
    {
        "name": "check_coverage",
        "description": "Check insurance coverage for a medical procedure",
        "input_schema": {
            "type": "object",
            "properties": {
                "procedure": {
                    "type": "string",
                    "description": "Name of the medical procedure"
                }
            },
            "required": ["procedure"]
        }
    }
]
def run_agent(user_message):
    print(f"\nUser: {user_message}")
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # If Claude is done and has a final answer
        if response.stop_reason == "end_turn":
            final_response = response.content[0].text
            print(f"\nAgent: {final_response}")
            return final_response
        
        # If Claude wants to use a tool
        if response.stop_reason == "tool_use":
            # Add assistant's response to messages
            messages.append({"role": "assistant", "content": response.content})
            
            # Process ALL tool calls in this response
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    print(f"\nAgent is using tool: {tool_name}")
                    print(f"With inputs: {tool_input}")
                    
                    if tool_name == "calculate_bmi":
                        tool_result = calculate_bmi(**tool_input)
                    elif tool_name == "get_patient_info":
                        tool_result = get_patient_info(**tool_input)
                    elif tool_name == "check_coverage":
                        tool_result = check_coverage(**tool_input)
                    
                    print(f"Tool result: {tool_result}")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(tool_result)
                    })
            
            # Add all tool results in one message
            messages.append({
                "role": "user",
                "content": tool_results
            })
# Test the agent
print("Healthcare AI Agent")
print("-" * 40)

run_agent("What is the BMI for someone who weighs 80kg and is 1.75m tall?")
run_agent("What medication is Sarah Johnson on?")
run_agent("Is an MRI covered by Elevance Health insurance?")
run_agent("Get me John Smith's info and also check if his annual wellness visit is covered")

