#This code is for gnerating output in terms of embeddings as it goes through each transformer blocks.

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

# Input embedding matrix
emb = model.get_input_embeddings().weight
print("=== INPUT EMBEDDING MATRIX ===")
print("shape:", emb.shape)
print(emb[:2, :8])

# Forward pass to collect hidden states
with torch.no_grad():
    out = model(
        input_ids=input_ids,
        attention_mask=attention,
        output_hidden_states=True,
        use_cache=False
    )

hidden_states = out.hidden_states

print("\n=== EMBEDDING OUTPUT (H0) ===")
print(hidden_states[0].shape)
print(hidden_states[0][0, :2, :8])

print("\n=== BLOCK OUTPUTS ===")
for i, h in enumerate(hidden_states[1:], start=1):
    print(f"\n--- Block {i} ---")
    print("shape:", h.shape)
    print(h[0, :2, :8])

# Output projection matrix
proj = model.get_output_embeddings().weight
print("\n=== OUTPUT PROJECTION MATRIX ===")
print("shape:", proj.shape)
print(proj[:2, :8])

# Generation
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

print("\n=== DECODED OUTPUT ===")
print(tokenizer.decode(gen[0], skip_special_tokens=True))
