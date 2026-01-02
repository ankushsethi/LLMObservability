#Forcing Intermediate block embeddings to human readbale form -> produces junk output

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = r"C:\Users\AnkushSethi\Desktop\Projects\vscode-workspace\tinyllama"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
    device_map="cpu",
    local_files_only=True,
    output_hidden_states=True
)

prompt = "Instruction: In one sentence, define the word 'Performance Testing'.\n\nAnswer:"

enc = tokenizer(prompt, return_tensors="pt")
input_ids = enc["input_ids"]
attention = enc["attention_mask"]

with torch.no_grad():
    out = model(
        input_ids=input_ids,
        attention_mask=attention,
        output_hidden_states=True,
        use_cache=False
    )

hidden_states = out.hidden_states
proj = model.get_output_embeddings().weight

print("=== INTERMEDIATE DECODED OUTPUTS ===")

for i, h in enumerate(hidden_states):
    hs = h[0]                      # [seq_len, dim]
    logits = hs @ proj.T           # [seq_len, vocab]
    ids = torch.argmax(logits, dim=-1)  
    text = tokenizer.decode(ids, skip_special_tokens=True)

    print(f"\n--- BLOCK {i} ---")
    print(text)

print("\n=== FINAL MODEL OUTPUT ===")
gen = model.generate(
    input_ids,
    attention_mask=attention,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.6,
    top_p=0.92,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(gen[0], skip_special_tokens=True))
