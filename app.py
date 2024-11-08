from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
hf_token = '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'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=hf_token)

# Set up the device to use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Assuming FAISS index is already built and loaded
# faiss_index = ... (load or create your FAISS index here)

# Function to retrieve relevant travel information
def retrieve_travel_info(query):
    # Placeholder retrieval function
    example_retrieved_info = [
        "Popular destinations in Italy include Rome, Venice, and Florence.",
        "The best time to visit Japan is from March to May and September to November.",
        "Paris is known for its iconic Eiffel Tower and art museums."
    ]
    return random.choice(example_retrieved_info)

# Function to generate chatbot response
def generate_response(conversation_history):
    inputs = tokenizer(conversation_history, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=512, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_response = response.split("Bot:")[-1].strip()
    return bot_response

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']
    
    # Retrieve relevant travel information
    relevant_info = retrieve_travel_info(user_input)
    
    # Prepare conversation history
    conversation_history = f"User: {user_input}\nTravel Info: {relevant_info}\nBot:"
    
    # Generate bot response
    bot_response = generate_response(conversation_history)
    
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)