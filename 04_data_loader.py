import torch  
import tiktoken
from torch.utils.data import Dataset, DataLoader    
#The Dataset
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        #Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        #use sliding window to chunk the book into overlapping sequences of max_length 
        # stride is the number of tokens to move the window forward for the next chunk, which is 1 in this case
        #stride 1 would give it lot more training examples to learn from and the model may overfit, meaning memorizes the text. but when stride = max_length, every token is used once as input with no repetition, this is used to avoid overlap
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            # since we are working with PyTorch, we need to convert the input and target chunks into tensors. It doesnt accept list of integers, so we convert them into tensors using torch.tensor() method.
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
            return len(self.input_ids)

    #DataLoader class in PyTorch uses the __getitem__ method to retrieve a single data point from the dataset. This method takes an index idx as input and returns the corresponding input and target tensors from the input_ids and target_ids lists. The DataLoader will call this method for each index in the dataset, allowing it to efficiently load batches of data during training.
    def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]
    
#The DataLoader
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
        #batch_size is how many cpu process u would like to run parallely, stride is how many previous tokens to skip to create the next, num_workers is number of cpu threads that we can run simultaneously to load the data, drop_last is whether to drop the last batch if it is smaller than the batch size, shuffle is whether to shuffle the data before creating batches
        #Intialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

        #Create the dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        #Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        #will see the getitem method in dataset
        # batch size is like the number of chunks the model will see at once and then update its parameters
    return dataloader
    
    
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter) # gives the next batch likewise
print(first_batch)
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.decode(first_batch[0][0].tolist()))   # first input chunk as text 
print(tokenizer.decode(first_batch[1][0].tolist()))   # its target as text
#iter() turns the DataLoader into an iterator, like C# IEnumerator
#since max length is 4 each tensor has 4 token IDs.
# input:   "THE"   " VER"   "D"    "ICT"
# target:  " VER"  "D"      "ICT"  "\n"