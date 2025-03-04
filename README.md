# Career Mentor Chatbot

This project demonstrates a Career Mentor Chatbot that leverages a fine-tuned large language model (Mistral-7B) to provide career advice and answer career-related queries. We fine-tuned a gated base model (Mistral-7B) using LoRA (Low-Rank Adaptation) on a curated career Q&A dataset, and built a Flask backend to serve the model. **Our React-based frontend is still under development.**

### Check out the model files here on Hugging Face:
https://huggingface.co/sp4rsh/career_mentor_finetuned_new/tree/main

## Overview

The goal of this project is to create a useful chatbot that helps users with questions such as:
- How to switch careers (e.g., from marketing to data science)
- How to start a career in AI research
- What companies hire Robotics Engineers
- Salary insights for various roles (e.g., Salesforce Administrator)

We achieved this by:
- **Fine-tuning a large pre-trained model** using LoRA for parameter-efficient adaptation.
- **Testing the model** in a Jupyter Notebook (see `Career_Mentor_V2.ipynb`) with sample prompts.
- **Building a Flask backend** (`app.py`) that loads the model and exposes a `/generate` API endpoint.
- **Starting development on a React frontend** to provide a user-friendly chat interface.

## Fine-Tuning Process

1. **Dataset Preparation:**  
   We curated a dataset of career Q&A pairs. Each entry was formatted as:
   - **Prompt:**  
     `Question: <your question here>\nAnswer:`
   - **Completion:**  
     The corresponding answer.
   
2. **Model Adaptation using LoRA:**  
   We applied LoRA for parameter-efficient fine-tuning on the base model (Mistral-7B). This process allows us to adapt the model with minimal additional storage, as only the adapter weights are saved.

3. **Testing in Jupyter Notebook:**  
   In `Career_Mentor_V2.ipynb`, we loaded the tokenizer and the fine-tuned model, then tested it using several prompts:
   ```python
   prompts = [
       "Question: How do I switch from marketing to data science?\nAnswer:",
       "Question: How can I begin working as an AI Researcher?\nAnswer:",
       "Question: What kind of companies hire Robotics Engineers?\nAnswer:",
       "Question: What is the average salary for a Salesforce Administrator?\nAnswer:"
   ]

   for prompt in prompts:
       inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
       outputs = model.generate(**inputs, max_new_tokens=100)
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)

       print(f"Prompt:\n{prompt}")
       print(f"Response:\n{response}")
       print("="*50)
   ```

# Backend

- The Flask backend (app.py) is responsible for:
	•	Loading the Model:
- It loads the base model in 4-bit quantization and applies the LoRA adapter.
	•	Providing an API Endpoint:
- A POST endpoint /generate accepts a JSON payload with a career question and returns a generated response.
	•	CORS Support:
-The backend uses flask-cors to allow cross-origin requests from the React frontend.

- Example usage (via cURL):
```
curl -X POST -H "Content-Type: application/json" \
     -d '{"prompt": "How do I switch from marketing to data science?"}' \
     http://localhost:5000/generate
```

# Frontend

- The React frontend is under development and will be available soon.