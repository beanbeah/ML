# Detecting LLM-generated Content.

#### You:
Solve the following machine learning task using Huggingface transformers. 

In a world where information is power, the evil APOCALYPSE organization has harnessed the might of large language models (LLMs) to spread fake news and manipulate public opinion. These nefarious actors have weaponized cutting-edge AI technology to undermine trust in legitimate sources of information and sow discord among the population.

But there is hope: a team of dedicated researchers and data scientists are working tirelessly to build a machine learning model that can detect LLM-generated content and flag it as potentially unreliable.

This cutting-edge technology analyzes not only the content of the text, but also the temperature associated with it. LLM-generated text tends to have a distinct temperature signature, which the model can use to distinguish it from genuine human-generated content.

You are given Base64 encoded sentences and their associated temperatures in train.csv but we are missing the temperatures for the Base64 encoded sentences in test.csv. Help us build a model to find the temperatures so we can stand up to the APOCALYPSE organization and their campaign of misinformation.

You will have to replace the '-1' placeholder with your predicted temperature and submit the submission.csv file to the graders listed below. Do not modify the file in any other way.


#### ChatGPT:
To tackle this problem, you can follow these steps:

1. Install Huggingface transformers library and other dependencies.
2. Load and preprocess the data.
3. Fine-tune a pre-trained model on your dataset.
4. Make predictions on the test dataset.
5. Generate the submission.csv file.

### Step 1: Install Huggingface transformers library and other dependencies

```bash
pip install transformers
pip install pandas
pip install torch
```

### Step 2: Load and preprocess the data

```python
import pandas as pd
from base64 import b64decode

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Decode the Base64 encoded sentences
train_data['decoded_text'] = train_data['base64_sentence'].apply(lambda x: b64decode(x).decode('utf-8'))
test_data['decoded_text'] = test_data['base64_sentence'].apply(lambda x: b64decode(x).decode('utf-8'))
```

### Step 3: Fine-tune a pre-trained model on your dataset

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)

# Tokenize the dataset
train_encodings = tokenizer(train_data['decoded_text'].tolist(), truncation=True, padding=True)

# Create a PyTorch dataset from the encodings and temperatures
import torch

class TempDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, temperatures):
        self.encodings = encodings
        self.temperatures = temperatures

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.temperatures[idx])
        return item

    def __len__(self):
        return len(self.temperatures)

train_dataset = TempDataset(train_encodings, train_data['temperature'].tolist())

# Set up the Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()
```

### Step 4: Make predictions on the test dataset

```python
test_encodings = tokenizer(test_data['decoded_text'].tolist(), truncation=True, padding=True)
test_dataset = TempDataset(test_encodings, [-1]*len(test_data))

# Predict temperatures for the test dataset
predictions = trainer.predict(test_dataset)
test_data['temperature'] = predictions.predictions.flatten()
```

### Step 5: Generate the submission.csv file

```python
submission = test_data[['id', 'temperature']]
submission.to_csv('submission.csv', index=False)
```

Now, you can submit the generated `submission.csv` file to the graders as specified.
