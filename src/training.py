import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import DigitDataset
from src.model import CNN_net

train_csv = '/Users/omama/Documents/Portfolio/digit_recognition/digit-recognizer/digitrecgog/data/train.csv'
train_df = DigitDataset(train_csv, has_labels=True)
# Create DataLoader
trainloader = torch.utils.data.DataLoader(train_df, batch_size=8, shuffle=True, num_workers=2)

model = CNN_net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}')
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}')
