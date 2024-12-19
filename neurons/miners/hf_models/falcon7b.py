from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("ybelkada/falcon-7b-sharded-bf16")
model = PeftModel.from_pretrained(base_model, "kasperius/falcon-7b-sharded-bf16-finetuned-html-code-generation-the-css-v2")

def generate_html_from_text(prompt, max_length=1024 * 10, num_return_sequences=1):
    """
    Generate text from a prompt using the Falcon-7B model
    
    Args:
        prompt (str): Input text prompt
        max_length (int): Maximum length of generated text
        num_return_sequences (int): Number of sequences to generate
        
    Returns:
        str: Generated text response
    """
    inputs = model.tokenizer(prompt, return_tensors="pt", padding=True)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=model.config.eos_token_id,
        do_sample=True
    )
    
    response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print(generate_html_from_text("Write a simple HTML page with a header, a paragraph, and a footer."))