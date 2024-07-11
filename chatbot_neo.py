from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # or "EleutherAI/gpt-neo-2.7B" if you have enough resources
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = inputs.ne(tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else None

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chat_with_bot(user_input):
    response = generate_response(user_input)
    return response
