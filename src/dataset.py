import torch
from torch.utils.data import Dataset
import pandas as pd


'''

This class is designed to handle our dataset, which is stored in a CSV file, and make it compatible with PyTorch's DataLoader

'''

class DigitDataset(Dataset):
    def __init__(self, csv_file, has_labels=True, transform=None):
        self.data = pd.read_csv(csv_file)
        self.has_labels = has_labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.has_labels:
            label = self.data.iloc[idx, 0]
            image = self.data.iloc[idx, 1:].values.astype('float32')
        else:
            label = -1  # Dummy label for test data
            image = self.data.iloc[idx].values.astype('float32')
        
        if len(image) != 784:
            raise ValueError(f"Expected 784 pixels but got {len(image)} pixels.")
        
        image = image.reshape(28, 28)  # Correctly reshape the image to 28x28
        image = torch.tensor(image).unsqueeze(0) / 255.0  # Add a channel dimension
        
        # Flatten the image for feeding into a fully connected layer
        #image = image.view(-1, 28*28)

        if self.has_labels:
            label = torch.tensor(label, dtype=torch.long)
            if self.transform:
                image = self.transform(image)
            return image, label
        else:
            if self.transform:
                image = self.transform(image)
            return image




# class DigitDataset(Dataset):
#     def __init__(self, csv_file, transform=None):
#         self.data = pd.read_csv(csv_file)
#         self.transform = transform
 
#  # Returns the number of samples in the dataset.       
#     def __len__(self):
#         return len(self.data) # Returns the number of rows in the DataFrame.
 
#  # Retrieves a sample from the dataset.   
#     def __getitem__(self, index):
#         label = self.data.iloc[index, 0] # Extracts the label (digit) from the first column of the DataFrame.
#         image = self.data.iloc[index, 1:].values.astype('float32') # Extracts the pixel values, converts them to a float32 numpy array, and reshapes it to a 28x28 image.
#         if len(image) != 784:
#             raise ValueError(f"Expected 784 pixels but got {len(image)} pixels.")
#         image = image.reshape(28, 28)
        
#         image = torch.tensor(image).unsqueeze(0) / 255.0 # Converts the numpy array to a PyTorch tensor and normalizes the pixel values to the range [0, 1]. The unsqueeze(0) adds a channel dimension, resulting in a shape of [1, 28, 28].
#         label = torch.tensor(label, dtype=torch.long) # Converts the label to a PyTorch tensor of type long (which is required for classification).
        
#         if self.transform: 
#             image = self.transform(image) # If a transform function is provided, it applies the transform to the image.
        
#         return image, label
    

