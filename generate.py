import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import getTransformerModel
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_DIM = 320
MAX_CONTEXT_LEN = 128
NUM_OF_HEADS = 5
DROPOUT = 0.2
NUM_OF_BLOCKS = 3
HEAD_SIZE = 320
PAD_TOKEN = 0
BATCH_SIZE = 32
SRC_LEN = 10
TRG_LEN = 15

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
VOCAB_SIZE = tokenizer.vocab_size
VOCAB_SIZE_SRC = VOCAB_SIZE
VOCAB_SIZE_TRG = VOCAB_SIZE


model = getTransformerModel(DEVICE, EMBEDDING_DIM, MAX_CONTEXT_LEN,
                            NUM_OF_HEADS, DROPOUT, NUM_OF_BLOCKS,
                            HEAD_SIZE, VOCAB_SIZE_TRG, VOCAB_SIZE_SRC, PAD_TOKEN).to(DEVICE)




def generate(model, src, tokenizer, MAX_CONTEXT_LEN, DEVICE):
    src = tokenizer(src, padding='max_length', truncation=True, max_length=MAX_CONTEXT_LEN, return_tensors="pt")
    src_mask = src["attention_mask"][0]
    src = src["input_ids"][0]
    src_mask = torch.tensor(src_mask).unsqueeze(0).to(DEVICE)
    src = torch.tensor(src).unsqueeze(0).to(DEVICE)



    output = [2]
    output_mask = [1]
    for i in range(MAX_CONTEXT_LEN - 1):
        out = model(src, torch.tensor(output).unsqueeze(0).to(DEVICE), src_mask, torch.tensor(output_mask).unsqueeze(0).to(DEVICE))
        most_probable_token = torch.argmax(out, dim=2)[0][i]
        output.append(most_probable_token)
        output_mask.append(1)

    print(tokenizer.decode(output, skip_special_tokens=True))


model.load_state_dict(torch.load('saved_model.pth'))
generate(model, "My name is John", tokenizer, MAX_CONTEXT_LEN, DEVICE)


