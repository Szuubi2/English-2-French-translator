import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import getTransformerModel
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
from generate import generate


print(torch.version.cuda)
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

# Downloads to C:\Users\<YourUsername>\.cache\huggingface\datasets\
dataset = load_dataset("wmt14", "fr-en")
train_data = dataset['train'].select(range(20000))
val_data = dataset['validation']

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        src = self.data[i]["translation"]["en"]
        trg = "<unk>" + self.data[i]["translation"]["fr"]

        src = tokenizer(src, padding='max_length', truncation=True, max_length=MAX_CONTEXT_LEN, return_tensors="pt")
        trg = tokenizer(trg, padding='max_length', truncation=True, max_length=MAX_CONTEXT_LEN, return_tensors="pt")

        src_mask = src["attention_mask"][0]
        trg_mask = trg["attention_mask"][0]

        src = src["input_ids"][0]
        trg = trg["input_ids"][0]

        return src, trg, src_mask, trg_mask
    

train_dataset = Dataset(train_data)
val_dataset = Dataset(val_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# inputs: src, trg, src_mask, trg_mask  outputs: out (B, T, VOCAB_SIZE_TRG)
model = getTransformerModel(DEVICE, EMBEDDING_DIM, MAX_CONTEXT_LEN,
                            NUM_OF_HEADS, DROPOUT, NUM_OF_BLOCKS,
                            HEAD_SIZE, VOCAB_SIZE_TRG, VOCAB_SIZE_SRC, PAD_TOKEN).to(DEVICE)

LEARNING_RATE = 3e-4
NUM_OF_EPOCHS = 20

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

try:
    model.load_state_dict(torch.load('saved_model.pth'))
    print('Loaded saved model.')
except FileNotFoundError:
    print('No saved model found, training from scratch.')


model.train()
for epoch in range(NUM_OF_EPOCHS):
    for  i, (src, trg, src_mask, trg_mask) in enumerate(iter(train_loader)):
        y = trg[:, 1:].to(DEVICE)
        src, trg, src_mask, trg_mask = src.to(DEVICE), trg[:, :-1].to(DEVICE), src_mask.to(DEVICE), trg_mask[:, :-1].to(DEVICE)
        out = model(src, trg, src_mask, trg_mask)

        B, T, C = out.size()
        loss = criterion(out.reshape(B*T, C), y.reshape(B*T))
        if i%10==0:
            print(f'Iter {i} Loss: {loss.item():.4f}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'saved_model.pth')


print('Training finished.')



