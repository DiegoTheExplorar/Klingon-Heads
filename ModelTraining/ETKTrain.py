import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import accelerate

# Load dataset
data = pd.read_csv('ModelTraining/English_To_Klingon.csv')

# Append <BOS> and <EOS> tags to the Klingon sentences
data['klingon'] = data['klingon'].apply(lambda x: '<BOS> ' + x + ' <EOS>')

# Separate the sentences
english_sentences = data['english'].values
klingon_sentences = data['klingon'].values

# Split data into training and testing sets. An 80 - 20 split is used here
english_train, english_test, klingon_train, klingon_test = train_test_split(
    english_sentences, klingon_sentences, test_size=0.2, random_state=42)

# Create a DataFrame for training and testing
train_df = pd.DataFrame({'translation': english_train, 'target': klingon_train})
test_df = pd.DataFrame({'translation': english_test, 'target': klingon_test})

# Convert DataFrame to Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')


def preprocess_function(examples):
    inputs = [ex for ex in examples['translation']]
    targets = [ex for ex in examples['target']]
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)

    # Use text_target for tokenizing the target labels
    labels = tokenizer(text_target=targets, max_length=128, padding="max_length", truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Load the model
model = T5ForConditionalGeneration.from_pretrained('t5-base')

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=7,              # total number of training epochs
    per_device_train_batch_size=4,   # batch size for training
    per_device_eval_batch_size=4,    # batch size for evaluation
    eval_strategy="epoch",
    logging_dir='./logs',            # directory for storing logs
    save_total_limit=2,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_train,       # training dataset
    eval_dataset=tokenized_test          # evaluation dataset
)

trainer.train()
model.save_pretrained('/ModelTraining/ETK_t5base_e7')
tokenizer.save_pretrained('/ModelTraining/ETK_t5base_e7')

## Model was too large to be uploaded to git. Find it here https://huggingface.co/TechRaj/ETK_t5base_e7