from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from your frontend

# Set up model variables
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Base model (gated, so you may need to add an auth token later)
LORA_REPO = "sp4rsh/career_mentor_finetuned_new"  # Your LoRA adapter repository name

# Force CPU usage since we're having MPS memory issues
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Global variables for model components
model = None
tokenizer = None

try:
    logger.info("Loading configuration...")
    peft_config = PeftConfig.from_pretrained(LORA_REPO)

    logger.info("Loading base model...")
    # Add memory optimization settings
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map={"": device},
        torch_dtype=torch.float32,  # Use float32 instead of float16
        low_cpu_mem_usage=True,     # Enable memory optimization
        offload_folder="offload",   # Enable model offloading
        trust_remote_code=True
    )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_REPO,
        adapter_name="default",
        is_trainable=False,
        torch_dtype=torch.float32  # Use float32 instead of float16
    )

    # Enable CPU offloading
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully!")

except Exception as e:
    logger.error(f"Error during model loading: {str(e)}")
    raise

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Server is running",
        "model_loaded": model is not None and tokenizer is not None,
        "usage": {
            "endpoint": "/generate",
            "method": "POST",
            "body": {"prompt": "Your career question here"},
            "example_curl": """curl -X POST http://127.0.0.1:5000/generate -H "Content-Type: application/json" -d '{"prompt":"What career path should I choose if I enjoy programming?"}'"""
        }
    })

@app.route("/generate", methods=["POST"])
def generate():
    try:
        if not model or not tokenizer:
            logger.error("Model or tokenizer not properly loaded")
            return jsonify({"error": "Model or tokenizer not properly loaded"}), 500

        data = request.get_json()
        if not data:
            logger.warning("No JSON data provided in request")
            return jsonify({"error": "No JSON data provided"}), 400
        
        prompt = data.get("prompt")
        if not prompt:
            logger.warning("No prompt provided in request")
            return jsonify({"error": "No prompt provided in request"}), 400
        
        logger.info(f"Received prompt: {prompt}")
        
        # Add memory optimization for inference
        start_time = time.time()
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Add parameters for more control over generation
            output = model.generate(
                **inputs,
                max_new_tokens=256,  # Reduced for better performance on CPU
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                num_return_sequences=1,
            )
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        logger.info(f"Generated response: {answer[:100]}...")  # Log just the beginning
        
        return jsonify({
            "response": answer,
            "generation_time_seconds": generation_time
        })
    
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Add host parameter and remove debug mode
    logger.info("Starting Flask server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, use_reloader=False)