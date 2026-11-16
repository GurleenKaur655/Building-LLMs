import urllib.request
import ssl
import zipfile
import os
from pathlib import Path
import importlib
import numpy as np

# to avoid loading the whole file 23
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"]) # the last layer at the very end of the architecture, and here we just use the same emdbedding tokens from the top which is called weight tying its done to prevent more computational costs and better optimization


mod = importlib.import_module("19_evaluatingllmperformance")

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. skipping download and extraction :)")
        return 
    #create an unverified SSL context
    ssl_context = ssl._create_unverified_context()

    #Downloading the file 
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    #unzipping the file 
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    #add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    
import pandas as pd

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
# makes it in 2 columns and as many rows as the obtained data
# 5572 rows * 2 columns

print(df["Label"].value_counts())
# Label
# ham     4825
# spam     747
# Name: count, dtype: int64
# so their is huge difference between the learning data between spam and no spam, hence for simplicity and because we prefer a small dataset for educational puposes, we would make the dataset in a way it contains 747 instances from each class.

def create_balanced_dataset(df):
    #count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    #randomly shape "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    #combine the hame "subset" with the "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
# Label
# ham     747
# spam    747
# Name: count, dtype: int64

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1}) # this is similar to converting the text into token IDs.. instead of using the GPT Vocabulary which consists of more than 50,000 words, we are dealing with only two token IDs: so 0 and 1 likewise

# we create a random_split function to split the dataset into 3 parts : 70% for training, 10% for validation, and 20% for testing
#these ratios are common in machine learning to train, adjust, and evaluate models
#these ratios are common in machine learning to train, adjust and evaluate models

def random_split(df, train_frac, validation_frac):
    #shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True) #frac 1 means 100% of the rows and .reset_index means renumbers the rows as well after shuffling

    #calculate split indices 
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # split the dataframe 
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]# practice ecam to check the score and change how we study 
    test_df = df[validation_end:] # one honest test

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

print(len(train_df))
print(len(validation_df))
print(len(test_df))
# 1045
# 149
# 300

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None) # dont save the row numbers into the CSV file, as the pandas have the indexing we dont need to save it 

#SpamDataset Class : identifies the longest sequence in the trainin dataset, encodes the text messages and ensures that all other sequences are padded with a padding token to match the length of the longest sequence

import torch 
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        #Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length() 
        else: 
            self.max_length = max_length 

            #Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ] # we need it for the validation and test sets because their max length we will be based off of the training max_length so anything longer than that would truncated

        #Pad sequences to the longest sequence 
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) # BODMAS rule
            for encoded_text in self.encoded_texts
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index] # special python message its what runs when we write train_dataset[5]
        label = self.data.iloc[index]["Label"]
        #.iloc[index] the row at this position and the label column so it get 0 or 1 for spam or ham
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0 
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length  > max_length: 
                max_length = encoded_length
        return max_length
    
import tiktoken 

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None, 
    tokenizer=tokenizer 
)

print(train_dataset.max_length)
# 120, which means the longest sequence contains no more than 120 tokens and our model can handle sequences of up to 1024 tokens given its context length limit

val_dataset = SpamDataset(
   csv_file="validation.csv",
   max_length=train_dataset.max_length, 
   tokenizer=tokenizer  
)

test_dataset = SpamDataset(
   csv_file="test.csv",
   max_length=train_dataset.max_length, 
   tokenizer=tokenizer  
)

print(test_dataset.max_length) # 120

# using the datasets as inputs, we can instantiate the data loaders similarly to what we did earlier
# However in this case, the targets reprsent class labels rather than the next tokens in the text

# for instance, choosing a batch size of 8, each batch will consist of 8 training examples of length 120 and corresponding target

from torch.utils.data import DataLoader

num_workers = 0 
batch_size = 8

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True, 
    num_workers=num_workers,
    drop_last=True
)

val_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True, 
    num_workers=num_workers,
    drop_last=True
)

test_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True, 
    num_workers=num_workers,
    drop_last=True
)

print("Train loader:")
for input_batch, target_batch in train_loader:
    pass # do nothing

print("Input batch dimesions:", input_batch.shape)
print("Label batch dimesions:", target_batch.shape)

# Input batch dimesions: torch.Size([8, 120])
# Label batch dimesions: torch.Size([8])

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} testing batches")
#130, 19, 38, [130, 8, 120] [130, 8]


CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257, # vocabulary size 
    "context_length": 1024, # context length
    "drop_rate":0.0, # dropout rate 
    "qkv_bias": True # query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
 
assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context"
    f"length {BASE_CONFIG['context_length']}. Reinitialize data set with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

from gpt_download3 import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = mod.GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()


# we could techincally use a single output node since we are dealing with a binary classification task. However, this would require modifying the loss function. Therefore, we choose a more general approach where the number of ouput nodes matches the number of classes. 

# for example a 3-class problem, such as classifying articles as "Technology", "Sports" or "Politics", we would use three output nodes and so forth.

# print(model)

# to get model ready for classification-finetuning, we will first freeze the model, meaning making all layers non-trainable

for param in model.parameters():
    param.requires_grad = False


# replace the output layer(model.out_head), which originally maps the layer inputs to 50,257 dimensions

torch.manual_seed(123)

num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes) # making the 50257 to be 2 
# since we changes the model.out_head dimension it automatically gets to true stage for gradient

for param in model.trf_blocks[-1].parameters(): #-1 means the last block
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True # last norm layer


inputs = tokenizer.encode("Do u have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs", inputs)
print("Inputs dimensions:", inputs.shape)
# Inputs tensor([[5211,  334,  423,  640]])
# Inputs dimensions: torch.Size([1, 4])

with torch.no_grad():
    outputs = model(inputs)

print("Outputs:\n", outputs)
print("ouputs dimensions:", outputs.shape)
# Outputs:
#  tensor([[[-1.5854,  0.9904],
#          [-4.0911,  6.4405],
#          [-2.4458,  5.7587],
#          [-4.1157,  4.6326]]])
# ouputs dimensions: torch.Size([1, 4, 2])

print("last output token:", outputs[:, -1, :])
# last output token: tensor([[-4.1157,  4.6326]])