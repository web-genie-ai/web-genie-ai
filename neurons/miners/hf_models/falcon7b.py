import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Loading original model
model_name = "ybelkada/falcon-7b-sharded-bf16"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
PEFT_MODEL = "PrincySinghal991/falcon-7b-sharded-bf16-finetuned-html-code-generation"
# PEFT_MODEL = "kasperius/falcon-7b-sharded-bf16-finetuned-html-code-generation-the-css-only"

peft_model = AutoModelForCausalLM.from_pretrained(
    PEFT_MODEL,
    quantization_config=bnb_config,
    device_map="auto",  # Let the transformers library handle device placement
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use mixed precision to reduce memory usage
    low_cpu_mem_usage=True
)

# Load tokenizer
peft_tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL, trust_remote_code=True)
peft_tokenizer.pad_token = peft_tokenizer.eos_token
 
def generate_html_from_text(prompt):
    # Tokenize and generate with the PEFT model
    peft_encoding = peft_tokenizer(prompt, return_tensors="pt")
    peft_outputs = peft_model.generate(
        input_ids=peft_encoding["input_ids"].to(peft_model.device), 
        attention_mask=peft_encoding["attention_mask"].to(peft_model.device),
        max_length=2048, 
        pad_token_id=peft_tokenizer.eos_token_id,
        eos_token_id=peft_tokenizer.eos_token_id
    )
    peft_model_html = peft_tokenizer.decode(peft_outputs[0], skip_special_tokens=True)
    return peft_model_html[len(prompt):]
    
if __name__ == "__main__":

    # Example usage
    prompt="create a simple login page with html and css"
    print("=================")
    import time
    start_time = time.time()
    print(f"Prompt: {prompt}")
    html = generate_html_from_text(prompt)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("=================")
    print(html)
     
