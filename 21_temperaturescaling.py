import torch

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
)
next_token_logits2 = next_token_logits/0.1

probas = torch.softmax(next_token_logits, dim=0)
print(probas)
probas2 = torch.softmax(next_token_logits2, dim=0)
print(probas2)
next_token_id = torch.argmax(probas).item()

print(next_token_id)
print(inverse_vocab[next_token_id]) # foward

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id]) # forward
# the multinomial function samples the next token proportional to its probability score. In other words, "forward" is still most likely token and will be selected by multinomial most of the time but not all the time


def print_sampled_tokens(probas):
    torch.manual_seed(123) # manual seed for reproductibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)] # pick one of the tokens and make it 1000 times, its multinomial is random, weighted so not always the highest but this does have highest but not always
    sampled_ids = torch.bincount(torch.tensor(sample)) # counts the number of times it comes in 
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")
    
print_sampled_tokens(probas)

# 73 x closer
# 0 x every
# 0 x effort
# 582 x forward
# 2 x inches
# 0 x moves
# 0 x pizza
# 343 x toward

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature 
    return torch.softmax(scaled_logits, dim=0)


#Temperature values
temperatures = [1, 0.1, 5] # original, higher confidence, and lower confidence

#calculate scaled probabilites 
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]

