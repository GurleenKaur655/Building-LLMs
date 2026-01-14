with open("the-verdict.txt", "r", encoding="utf-8") as inputFile: 
    # r means we are reading this file 
    raw_text = inputFile.read()

print("Total number of characters:", len(raw_text)) # Prints the total number of characters in the text
print(raw_text[:99]) # Prints the first 99 characters of the text

# Goal is to tokenize the text into words

# will use re python library to help with tokenization, re = regular expressions
import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()] # command used to remove whitespace 
# scan from left to right, looking for whitespace characters (\s)
# whenever we find one, we split the string at that point
# the parentheses around \s means we want to keep the whitespace characters in the result
print(preprocessed[:20]) # print the first 20 tokens
print(len(preprocessed)) # print the total number of tokens


# Creating Token Ids

# creating vocabulary which is basically set of unique tokens
all_words = sorted(set(preprocessed)) # sorted makes in in alphabetical order ad set makes it unique
vocab_size = len(all_words)

print("Vocab size:", vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)} # enumerate gives an integer to each token

class SimpleTokenizerV1: # Rule based Tokenizer
    def __init__(self, vocab): # called when the instance of the class is created and vocab is mapping from token to token Ids
        # self is basically the instance of the class
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()} # we take string which is token (s in this case) and i which is integer (i.e. token Id) we reverse the mapping by doing i:s instead of s:i, so we flip the mapping 

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed] # This list comprehension converts each token in the preprocessed text into its corresponding integer ID by looking it up in the vocabulary dictionary.
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids]) # join method joins the list of strings into a single string with spaces in between as int_to_str return the token(string) which is value to the key which is integer which is token Id
        # replace spaces before the specified punctuations 
        text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text) # getting rid of spaces before punctuation
        return text
    
tokenizer = SimpleTokenizerV1(vocab)

text = """It's the last he painted, you know, 
            Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

text = tokenizer.decode(ids)
print(text)
    

