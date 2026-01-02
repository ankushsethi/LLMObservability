#This is default working code. No Trash.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = r"C:\Users\AnkushSethi\Desktop\Projects\vscode-workspace\tinyllama"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

print("Tokenizer:", tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32, #Controls numerical precision of the model’s weights.Float32 is the baseline, slowest, most stable choice on CPU.
    device_map="cpu", #CPU only
    local_files_only=True #Disable Remote fetch
)

print("Model:", model)

prompt = "Instruction: In one sentence, define the word 'Performance Testing'.\n\nAnswer:"

#The tokenizer converts raw text into numerical token IDs.
enc = tokenizer(prompt, return_tensors="pt") #returned dictionary containing all tensorized components of the input.
print("ENC:", enc)
input_ids = enc["input_ids"] #tensor of token indices.
print("Input IDs:", input_ids)
attention = enc["attention_mask"]

outputs = model.generate(
    input_ids, #sequence of integers representing vocabulary tokens
    attention_mask=attention, #sequence of 1s for valid tokens. No padding here, so every position is valid.
    max_new_tokens=100, #Hard cap on how many tokens the model is allowed to generate beyond the prompt.
    do_sample=True, #Next token is chosen based on probability rather than most likely one (greedy decoding).
    temperature=0.6, #Higher temperature = more random sampling. Lower temperature = more greedy sampling.
    top_p=0.92,
    repetition_penalty=1.1, #Penalty applied to tokens already generated. Higher values reduce loops and repeated phrases.
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

print("Outputs:", outputs)

generated = outputs[0]
gen_part = generated[input_ids.shape[1]:]

print("=== DECODED OUTPUT ===")
print(tokenizer.decode(generated, skip_special_tokens=True))

print("\n=== DEBUG ===")
print("input tokens:", input_ids.shape[1])
print("generated tokens:", gen_part.shape[0])
print("first 20 generated token ids:", gen_part.tolist()[:20])
