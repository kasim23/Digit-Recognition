import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.dataset import DigitDataset
from src.model import CNN_net
from src.training import model

test_csv = '/Users/omama/Documents/Portfolio/digit_recognition/digit-recognizer/digitrecgog/data/test.csv'
test_df = DigitDataset(test_csv, has_labels=False) 
# Create DataLoader
testloader = torch.utils.data.DataLoader(test_df, batch_size=8, shuffle=False, num_workers=2)


# Generate predictions for the test set
model.eval()
predictions = []
with torch.no_grad():
    for images in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

# Create a DataFrame with ImageId and Label columns
submission = pd.DataFrame({
    "ImageId": list(range(1, len(predictions) + 1)),
    "Label": predictions
})

# Save the DataFrame to a CSV file
submission.to_csv('submission_CNN.csv', index=False)

print('Submission file created successfully!')