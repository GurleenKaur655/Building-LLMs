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


#we can obtain the class label via the follwing code 
# probas = torch.softmax(outputs[:, -1, :], dim=-1)
# print(probas) # [0.0005, 0.9995]
# label = torch.argmax(probas)
# print("Class label:", label.item())
# class label 1 means its a spam
# using softmax here is optional since the largest ouputs direclty correspond to the highest probabilities 


# to determine the classification accuracy, we apply the argmax-based prediction code to all the examples in the dataset and calculate the proportion
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0 

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches: 
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:,-1,:] #logtis of last ouput token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item() #.sum() doesnt return a plain number, returns a tensor holding one number, .item() pulls that number out as a regular Python int
        else:
            break 
    return correct_predictions / num_examples 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

torch.manual_seed(123)

train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches = 10)

val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches = 10)

test_accuracy = calc_accuracy_loader(test_loader, model , device, num_batches =10)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Training accuracy: 46.25%
# Validation accuracy: 46.25%
# Test accuracy: 43.75%

# classification accuracy is not a differentiable function , so we use cross entropy loss as a proxy to maximize accuracy 


# finetuning the model on supervised data 
#Step 1: set the model to training mode
#2: reset loss gradients from previous batch iteration
#3: calculate loss gradients 
#4: update model weights using loss gradients
#5: new: track examples instead of tokens
#6: optional evaluation step 
#7: calculate accuracy after each epoch 
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :] # logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss 

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else: 
        # reduce the number of batches to match the total numebr of batches in the data loader 
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss/ num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # removes dropout
    with torch.no_grad(): # removes the gradient calculation because we are not training but evaluating
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter) # this = to eval_iter means it only checks after a few batches instead of a whole dataset, so u can call it often during training without slowing down much 
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):

    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop 
    for epoch in range(num_epochs):
        model.train() # set model to training mode 

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # reset loss gradients from previous batch iteration 
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # calculate loss gradients 
            optimizer.step() # update model weights using loss gradients 
            examples_seen += input_batch.shape[0] # New : track examples instead of tokens
            global_step += 1

            # Optional evaluation step 
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter 
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep{epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
        # calculate accuracy after each epoch 
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100: .2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100: .2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    return train_losses, val_losses, train_accs, val_accs, examples_seen


import time 
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr =5e-5, weight_decay=0.1)

num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs= num_epochs, eval_freq=50, eval_iter=5, 
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in { execution_time_minutes:.2f} minutes.")

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100: .2f}%")
print(f"Validation accuracy: {val_accuracy*100: .2f}%")
print(f"Test accuracy: {test_accuracy*100: .2f}%")

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    input_ids = input_ids[:min(max_length, supported_context_length)]

    #pad sequences to the longest seqeunce 
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    #model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :] # logits of the last output token 
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"


text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or $2000 award."
)

print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

text_2 = (
    "hey, just wanted to check if we are still on"
    "for dinner tonight?let me know"
)

print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

torch.save(model.state_dict(), "review_classifier.pth")

model_state_dict = torch.load("review_classifier.pth")
model.load_state_dict(model_state_dict)