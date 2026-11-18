from importlib import metadata
import tiktoken 

tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r", encoding="utf-8") as inputFile: 
    # r means we are reading this file 
    raw_text = inputFile.read()
enc_text = tokenizer.encode(raw_text) 
#uses a dictionary but returns a list #convert this text into token and then into token ids using a tokenizer
print(len(enc_text)) #5145

enc_sample = enc_text[50:] # is a list #skips first 50 tokens and takes the rest of the tokens

context_size = 4
# context_size of 4 means that the model is trained to look at a sequence of 4 words (or tokens) and to predict the next word in the sequence. 
# the input x is the first 4 tokens [1,2,3,4] and the target y is the next 4 tokens [2,3,4,5] but the input cant be [1,2,3,4,5] because the model is only looking at 4 tokens at a time
#the input could be less than or equal to the context size but cant be greater than the context size

x = enc_sample[:context_size] # enc_sample[4]
y = enc_sample[1:context_size+1]#[1:5] not including 5

print(f"x: {x}")
print(f"y:      {y}")

# processing the inputs along with the targets,which are the inputs shifted by one position, we can then create the next-word prediction tasks as follows: 
for i in range(1, context_size+1):
    context = enc_sample[:i]#i = 0
    desired = enc_sample[i]# i = 1

    print(context, "---->", desired)
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# the x and y are used to create the input-target pairs.





