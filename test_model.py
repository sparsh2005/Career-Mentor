import requests
import json
import time

# URL of your local Flask server
URL = "http://127.0.0.1:5000/generate"

def test_prompt(prompt):
    """Send a prompt to the model and print the response."""
    print(f"\n\033[1mTesting prompt:\033[0m {prompt}")
    print("-" * 50)
    
    # Prepare the data
    data = {"prompt": prompt}
    
    # Send the request
    start_time = time.time()
    try:
        response = requests.post(URL, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print(f"\033[92mSuccess!\033[0m Response received in {time.time() - start_time:.2f} seconds.")
            print("\n\033[1mModel Response:\033[0m")
            print(result["response"])
        else:
            print(f"\033[91mError {response.status_code}:\033[0m {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("\033[91mConnection Error:\033[0m Could not connect to the server.")
        print("Make sure the Flask server is running at http://127.0.0.1:5000")
    
    except Exception as e:
        print(f"\033[91mError:\033[0m {str(e)}")
    
    print("-" * 50)

# List of prompts to test
prompts = [
    "What career path should I choose if I enjoy programming?",
    "How do I transition from marketing to data science?",
    "What skills should I develop to become a successful product manager?",
    "I'm interested in AI and machine learning. What career options do I have?",
    "How do I know if entrepreneurship is right for me?"
]

# Test if the server is running
try:
    requests.get("http://127.0.0.1:5000/")
    print("\033[92mServer is running!\033[0m Let's test some prompts...")
except:
    print("\033[91mServer is not running!\033[0m Please start the Flask server first.")
    exit(1)

# Test each prompt
for prompt in prompts:
    test_prompt(prompt)
    # Wait a bit between requests to avoid overwhelming the CPU
    time.sleep(2)

print("\nAll tests completed!")