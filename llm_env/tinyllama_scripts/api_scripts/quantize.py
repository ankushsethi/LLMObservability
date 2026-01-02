from transformers import AutoModelForCausalLM, AutoTokenizer

#src = "meta-llama/Llama-3.2-3B-Instruct"
src = "llama3B"
dst = "llama32-3b-4bit"

tokenizer = AutoTokenizer.from_pretrained(src)

model = AutoModelForCausalLM.from_pretrained(
    src,
    load_in_4bit=True,
    device_map="cpu"
)

model.save_pretrained(dst)
tokenizer.save_pretrained(dst)