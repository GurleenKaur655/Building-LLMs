import urllib.request
import ssl
import zipfile
import os
from pathlib import Path

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
test_df.to_csv("test.csv", index=None)