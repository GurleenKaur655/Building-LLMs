from importlib import metadata
import tiktoken 

print("tiktoken version:", metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2") 
# this is a byte-pair encoding (BPE) tokenizer used by GPT-2 and GPT-3 models

text = ("Hello, do you like tea? <|endoftext|> In the sunlight terraces of someunknownPlace.")

# endoftext is a part of Gpt-2 tokenizer special tokens

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# english vocabulary has about 170000 to 200000 words but gpt-2 tokenizer has a vocabulary of only 50257 tokens becuase it uses BPE which breaks down words into subwords or smaller units

print(integers)

strings = tokenizer.decode(integers)
print(strings) 
# every tokenizer has encode and decode methods and tiktoken is no exception but here we use encode of tiktoken which is more advanced than our simple tokenizer because it uses BPE and we use a decoder which is not used by GPT-2 models but it is available for us to use

# GPT-2 and GPT-3 has vocabulary of 50257 tokens and endoftext is one has the largest token Id 50256 and 50257 is reserved for unknown tokens

integers = tokenizer.encode("Akwirw ier")
print(integers)

strings = tokenizer.decode(integers)
print(strings)

