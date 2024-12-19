import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = AutoModelForCausalLM.from_pretrained("ybelkada/falcon-7b-sharded-bf16")
MODEL = PeftModel.from_pretrained(
    BASE_MODEL, 
    "PrincySinghal991/falcon-7b-sharded-bf16-finetuned-html-code-generation"
).to(DEVICE)

TOKENIZER = AutoTokenizer.from_pretrained("ybelkada/falcon-7b-sharded-bf16")
TOKENIZER.pad_token = TOKENIZER.eos_token

def generate_html_from_text(prompt, max_length=4096, num_return_sequences=1):
    """
    Generate text from a prompt using the Falcon-7B model
    
    Args:
        prompt (str): Input text prompt
        max_length (int): Maximum length of generated text
        num_return_sequences (int): Number of sequences to generate
        
    Returns:
        str: Generated text response
    """ 
    input_ids = TOKENIZER(prompt, return_tensors="pt").input_ids.to(DEVICE)

    outputs = MODEL.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=MODEL.config.eos_token_id,
        do_sample=True
    )
    
    response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print(generate_html_from_text("Write a simple HTML page with a header, a paragraph, and a footer."))