from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = inputs.ne(tokenizer.eos_token_id).long()
    
    # Generate response with sampling
    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask,
        max_length=150, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,       # Enable sampling
        temperature=0.7,      # Adjust temperature
        top_k=50,             # Top-k sampling
        top_p=0.95            # Nucleus sampling
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chat_with_bot(user_input):
    response = generate_response(user_input)
    return response
