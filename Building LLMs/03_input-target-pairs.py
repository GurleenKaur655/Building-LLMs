from importlib import metadata
import tiktoken 

tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r", encoding="utf-8") as inputFile: 
    # r means we are reading this file 
    raw_text = inputFile.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4
# context_size of 4 means that the model is trained to look at a sequence of 4 words (or tokens) and to predict the next word in the sequence. 
# the input x is the first 4 tokens [1,2,3,4] and the target y is the next 4 tokens [2,3,4,5] but the input cant be [1,2,3,4,5] because the model is only looking at 4 tokens at a time

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")



