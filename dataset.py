# You are already familiar with the Model class. Now you will have to create your own dataset using torch.utils.data.Dataset. The required steps to implement a custom dataset are:
# Implement a class that inherits from Dataset.
# Call the super().__init__() method in the __init__. You should do all the preliminary stuff here (like reading the csv). You should also add a transforms argument to use in the __getitem__.
# Implement a __len__ method with the length of the dataset.
# Implement a __getitem__ method that returns a transformed sample and its label.

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform
        
        

    def __len__(self):
        return len(self.labels_df)
        


    def __getitem__(self, idx):
        suite_id, sample_id, code, value, character = self.labels_df.loc[idx, :]
        path = os.path.join(self.images_path, f"input_{suite_id}_{sample_id}_{code}.jpg")
        sample = Image.open(path)
        if self.transform:
            sample = self.transform(sample)

        return sample, code-1 #code goes from 1 to 5
    

   