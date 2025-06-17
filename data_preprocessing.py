# data_preprocessing.py

from datasets import load_dataset
from transformers import AutoTokenizer
import tensorflow as tf

def load_and_preprocess_dataset(model_name="bert-base-uncased", max_length=32, batch_size=32, intent_limit=None):
    """
    Load the CLINC150 dataset, encode the data using the specified tokenizer, and convert it to a TensorFlow Dataset. 
    Simultaneously remove the samples with intent = 42 from the train and validation data.
    """

    print("Loading dataset...")
    dataset = load_dataset("clinc_oos", "plus")
    train_data, val_data, test_data = dataset["train"], dataset["validation"], dataset["test"]

    def remove_intent_42(example):
        return example["intent"] != 42

    train_data = train_data.filter(remove_intent_42)
    val_data = val_data.filter(remove_intent_42)
    test_data = test_data.filter(remove_intent_42)

    if intent_limit is not None:
        def filter_by_label(example):
            return example["intent"] < intent_limit
        train_data = train_data.filter(filter_by_label)
        val_data = val_data.filter(filter_by_label)
        test_data = test_data.filter(filter_by_label)
        num_labels = intent_limit
    else:
        num_labels = dataset["train"].features["intent"].num_classes


    print(f"Dataset loaded. Total intents after filtering: {num_labels}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    print("Tokenizing...")
    train_data = train_data.map(tokenize_fn, batched=True)
    val_data = val_data.map(tokenize_fn, batched=True)
    test_data = test_data.map(tokenize_fn, batched=True)

    train_set = train_data.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="intent",
        shuffle=True,
        batch_size=batch_size
    )
    val_set = val_data.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="intent",
        shuffle=False,
        batch_size=batch_size
    )
    test_set = test_data.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="intent",
        shuffle=False,
        batch_size=batch_size
    )

    return train_set, val_set, test_set, num_labels, tokenizer
