import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate text with GPT-Neo
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to determine sentiment using GPT-Neo
def determine_sentiment(text):
    prompt = f"Text: {text}\nSentiment: "
    response = generate_response(prompt)
    # Extract sentiment from response
    if "positive" in response.lower():
        return "Positive"
    elif "negative" in response.lower():
        return "Negative"
    else:
        return "Neutral"

# Function to determine tone using GPT-Neo
def determine_tone(text):
    prompt = f"Text: {text}\nTone: "
    response = generate_response(prompt)
    # Extract tone from response
    tones = ["harsh", "cool", "composed"]
    for tone in tones:
        if tone in response.lower():
            return tone.capitalize()
    return "Neutral"

# Function to process CSV and add sentiment and tone columns
def process_csv(file):
    df = pd.read_csv(file)
    if 'sentence' not in df.columns:
        return "CSV file must have a 'sentence' column"
    
    # Limit to 10 sentences
    df = df.head(10)
    df['sentiment'] = df['sentence'].apply(determine_sentiment)
    df['tone'] = df['sentence'].apply(determine_tone)

    return df
