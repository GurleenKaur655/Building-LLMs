import torch 
import importlib
import tiktoken 

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024, 
    "emb_dim": 768,
    "n_heads": 12, 
    "n_layers": 12, 
    "drop_rate": 0.1,
    "qkv_bias": False
}

gptmodel = importlib.import_module("12_gptarchitecture")
GPTModel = gptmodel.GPTModel

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

def generate_text_sample(model, idx, max_new_tokens, context_size):
    #idx is (batch, n_tokens) array of indices in the current context
    #idx ia the input batch 
    #tensor([[6190, 3626, 6100, 345],
    #[6190, 1110, 6622, 257]]) idx is basically the tokens that we are giving to pass through the transformer and make predictions later 
    for _ in range(max_new_tokens):

        #crop the current context if it exceeds the supported context size
        #eg if LLM supports only 5 tokens and the context size is 10, then only the last 5 are used as context, like the tokens need to same as the context size
        idx_cond = idx[:, -context_size:]

        #get the predictions
        with torch.no_grad():
            logits = model(idx_cond) # shape (batch, context_length, vocab_size)

        #focus only on the last time step 
        #(batch, n_tokens, vocab_size)becomes (batch, vocab_size) because we take only the last one from each batch to make the prediction
        logits = logits[:, -1, :]

        #apply softmax to get probabilities 
        probas = torch.softmax(logits, dim=-1) # (batch , vocab_size)

        #get the idx of the vocab entry with the highest probability value 
        idx_next = torch.argmax(probas, dim =-1, keepdim=True) #(batch, 1)

        #Append sampled index to the running sequence 
        idx = torch.cat((idx, idx_next), dim=1) # (batch, n_tokens+1)
    return idx

tokenizer = tiktoken.get_encoding("gpt2") 
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape", encoded_tensor.shape)


model.eval() #we are putting the model in the evaluation mode passes some layers such as dropout layer or normalization layer 
out = generate_text_sample(
    model = model, 
    idx=encoded_tensor, 
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# decode wants [int, int, int, ...]. Without the squeeze you’d hand it [[int, int, ...]], a batch of one sequence, and it fails.
print(decoded_text)
# Hello, I am Featureiman Byeswickattribute argue

# it random because we havent trained the model yet and all those parameters are random right now, so far we just implemented the GPT architecture and initialized a GPT model instance with initial random weights.