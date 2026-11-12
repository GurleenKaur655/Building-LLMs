import torch 
import importlib
mod = importlib.import_module("19_evaluatingllmperformance")

vocab = {
    "closer": 0, 
    "every": 1, 
    "effort": 2, 
    "forward": 3, 
    "inches": 4, 
    "moves": 5, 
    "pizza": 6, 
    "toward": 7, 
    "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
) # kind of like values on what could be best outcome 

top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
#topk returns "k" the largest elements of the given "input"
#tensor along a given dimesion 
print("Top logits:", top_logits)
print("Top positions:", top_pos)
# top logits: tensor([6.7500, 6.2800, 4.5100])
# Top positions: tensor([3, 7, 0])

new_logits = torch.where(
    condition = next_token_logits < top_logits[-1], #the tokens not present in top_logits turn into inf
    input = torch.tensor(float("-inf")),
    other=next_token_logits 
)

print(new_logits)
#tensor([4.5100,    inf,    inf, 6.7500,    inf,    inf,    inf, 6.2800,    inf])

topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)
# tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    #for-loop is the same as before: get logits, and only focus on last time step 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # idx has shape(batch, tokens), for example (1, 300): 1 sentence thats 300 tokens long. # keep only the last context_size tokens (sliding window) so we never exceed the model's max length, basically the shape is (batch, min(tokens_so_far, context_size))

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] # pos1 = keep all prompts in the batch, pos2 = -1 take only the last token postion, 3rd pos = keep all 50,257 or whatver the vocab size is so the logits shape becomes(1, 50257)

        #New: Filter logits with the top_k sampling 
        if top_k is not None:
            #keep only top_k values 
            # topk returns two values value and indices(position of the top values), _ means i wont use that variable again ever, idc about it 
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1] #: = all rows, and -1 = last column
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        #Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # apply softmax to get probabilities 
            probs = torch.softmax(logits, dim=-1) # (batch_size, context_len)

            #sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1) #(batch_size, 1)

        #Otherwise same as before: get idx of the vocab entry with the highest logits value
        else: 
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) #(batch_size, 1)

        if idx_next == eos_id: # stop generating early if the end-of-sequence token is encountered and eos_id is specified
            break 

        #same as before: append sampled index to the running sequence 
        idx = torch.cat((idx, idx_next), dim=1) # (batch_size, num_tokens+1)
    return idx


torch.manual_seed(123)
model = mod.GPTModel(mod.GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=mod.device))  # load the trained weights
model.to(mod.device)
model.eval()

token_ids = generate(
    model=model, 
    idx=mod.text_to_token_ids("Every effort moves you", mod.tokenizer),
    max_new_tokens=15, 
    context_size=mod.GPT_CONFIG_124M["context_length"],
    top_k=25, 
    temperature=1.4
)

print("Ouput text:\n", mod.token_ids_to_text(token_ids, mod.tokenizer))
# Ouput text:
#  Every effort moves youEveryiliaralso stabbed OrleansAllowsean 52anche crime winter unbeaten quoteembedreportprint earning
# reason to get such input is that my trained model is in a different file which is pretraining.py and the it doesnt store the trained weights anywhere and hence it gives such input

# Ouput text: and now the text improved exponentially 
# Every effort moves you know he said, with a cheap genius--and that    = text from just training weights
#  Every effort moves you know; then lends it to paint's it happened how lucky chap, but