!huggingface-cli login --token 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

from helper_func import get_image_paths_labels
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

#Organize turing data into a dataframe
# base_folder = 'cats_ds'
# data_list = get_image_paths_labels(base_folder)
# df = pd.DataFrame(data_list)
# df.to_csv("cats_ds/metadata.csv")

dataset = load_dataset("imagefolder", data_dir="cats_ds",split="train")
labels = ['prim','rupe','notcat']

# You could create a mapping from string labels to integers
label_to_int = {label: index for index, label in enumerate(set(labels))}

# Now you can convert your string labels to integers using this mapping
int_labels = [label_to_int[label] for label in labels]

df = pd.DataFrame(dataset)

# Split the data into training and validation sets
train_dataset, val_dataset = train_test_split(df, test_size=0.1)  # Here, 10% is used as validation set

from datasets import Features, ClassLabel, Dataset, Image, DatasetDict

# Define features, specifying that 'labels' is a ClassLabel
features = Features({
    'image': Image(),
    'labels': ClassLabel(num_classes=3, names=['prim','rupe','notcat'])
})

# Create a Dataset object with these features
train_dataset = Dataset.from_dict({
    'image': train_dataset['image'], 
    'labels': train_dataset['labels']
}, features=features)

val_dataset = Dataset.from_dict({
    'image': val_dataset['image'], 
    'labels': val_dataset['labels']
}, features=features)

# Create a DatasetDict
dataset_dict = DatasetDict({ 
    'train': train_dataset,
    'validation': val_dataset
})

dataset_dict.push_to_hub("ChrisGuarino/cats")