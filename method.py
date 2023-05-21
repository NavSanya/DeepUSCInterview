import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling

# Step 1: Train the baseline model for LM and report perplexity

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the maximum sequence length and batch size
max_seq_length = 128
batch_size = 32

# Load the WikiText-2 dataset
train_dataset = TextDataset(tokenizer=tokenizer, file_path="wikitext-2/train.txt", block_size=max_seq_length)
valid_dataset = TextDataset(tokenizer=tokenizer, file_path="wikitext-2/valid.txt", block_size=max_seq_length)

# Create the data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create the data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=data_collator)

# Training loop for the baseline model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
num_epochs = 5

for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
    average_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} - Average Loss: {average_loss:.4f}")

# Evaluation on the validation set
model.eval()

total_loss = 0
num_tokens = 0

with torch.no_grad():
    for batch in valid_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        
        total_loss += loss.item()
        num_tokens += batch["attention_mask"].sum().item()

perplexity = torch.exp(total_loss / num_tokens)
print(f"Baseline Model Perplexity: {perplexity:.4f}")

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

# Step 2: Perform Dataset Distillation using SimCLR on WikiText-2 and extract 120 data points

# Load the WikiText-2 dataset
dataset = ImageFolder("wikitext-2/images", transform=transforms.ToTensor())

# Split the dataset into training and validation sets
train_dataset, valid_dataset = train_test_split(dataset, train_size=0.8, test_size=0.2, random_state=42)

# Set the batch size and number of training iterations
batch_size = 32
num_iterations = 1000

# Create the data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

# Implement SimCLR for Dataset Distillation
simclr_model = ...  # Replace with your implementation of SimCLR

# Training loop for SimCLR
for iteration in range(num_iterations):
    for images, _ in train_dataloader:
        # Apply augmentations to the images
        augmented_images = ...  # Replace with your augmentation code
        
        # Perform forward pass through the SimCLR model
        embeddings = simclr_model(augmented_images)
        
        # Store the embeddings for further processing
        
# Extract 120 data points based on similarity
distilled_data = ...  # Replace with your code to extract 120 data points from the embeddings

# Now you have the distilled data with 120 data points for further training

# Step 3: Train a new model from scratch on the distilled data points

# Split the distilled data into training and validation sets
train_data = distilled_data[:120]
valid_data = distilled_data[120:]

# Create the data loaders for the distilled data
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=data_collator)

# Train a new model from scratch using the distilled data
new_model = GPT2LMHeadModel.from_pretrained("gpt2")
new_model.resize_token_embeddings(len(tokenizer))
new_model.to(device)
new_model.train()

optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-5)
num_epochs = 5

for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        outputs = new_model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
    average_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} - Average Loss: {average_loss:.4f}")

# Evaluation on the validation set
new_model.eval()

total_loss = 0
num_tokens = 0

with torch.no_grad():
    for batch in valid_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = new_model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        
        total_loss += loss.item()
        num_tokens += batch["attention_mask"].sum().item()

perplexity = torch.exp(total_loss / num_tokens)
print(f"New Model Perplexity: {perplexity:.4f}")