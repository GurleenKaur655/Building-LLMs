import torch 
import importlib
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, #shorten for the trianing simplicity
    "emb_dim": 768,
    "n_heads": 12, 
    "n_layers": 12, 
    "drop_rate": 0.1,
    "qkv_bias": False
}
gptmodel = importlib.import_module("12_gptarchitecture")
GPTModel = gptmodel.GPTModel

generate_text_sample = importlib.import_module("17_predictnexttoken")
generate_text_sample = generate_text_sample.generate_text_sample

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval() # disable dropout during inference

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension 
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_sample(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10, 
    context_size=GPT_CONFIG_124M["context_length"]
)
token_list = token_ids_to_text(token_ids, tokenizer)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
print(len(token_ids[0])) #14 4+10

#Every effort moves you rentingetic wasnم refres RexMeCHicular stren

inputs = torch.tensor([[16833, 3626, 6100], #["every effort moves",
                       [40, 1107, 588]]) # "I really like"]

targets = torch.tensor([[3626, 6100, 345], #["effort moves you",
                        [1107, 588, 11311]]) # "really like chocolate"]

with torch.no_grad(): #disables gradient calculation
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1) # probability of each token in vocabulary
print(probas.shape) #shape: (batch_size, num_tokens, vocab_size)
# torch.Size([2, 3, 50257])

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)
# Token IDs:
#  tensor([[[16657],
#          [  339],
#          [42826]],

#         [[49906],
#          [29669],
#          [41751]]])

print(f"Targets batch 1:{token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# Targets batch 1: effort moves you
# Outputs batch 1:  Armed heNetflix
# its not a new prediction its basically just for every effort moves, so the target is effort moves you

text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]] # it gives its dimension batch = 0, position = 3, vocab(which is basically correct token ID at each position) 
print("Text 1:", target_probas_1)

text_idx = 1 
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Text 2:", target_probas_2)


log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

# tensor([ -9.5042, -10.3796, -11.3677, -11.4798,  -9.7764, -12.2561])

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
# tensor(-10.7940)

neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)
# tensor(10.7940)

logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()

print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)
# Flattened logits: torch.Size([6, 50257])
# Flattened targets: torch.Size([6])

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
# tensor(10.7940)


