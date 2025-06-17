# evaluate.py

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import tensorflow as tf
from sklearn.metrics import classification_report
from data_preprocessing import load_and_preprocess_dataset

def get_predictions(model, dataset):
    preds = []
    labels = []

    for batch in dataset:
        features, y_true = batch  # unpack
        inputs = {k: features[k] for k in ["input_ids", "attention_mask"]}
        logits = model(inputs, training=False).logits
        y_pred = tf.argmax(logits, axis=1).numpy()

        preds.extend(y_pred)
        labels.extend(y_true.numpy())
    return labels, preds


def evaluate_classification_metrics():
    print("Evaluating model performance...")

    _, _, test_ds, _, _ = load_and_preprocess_dataset(intent_limit=None)

    y_true, y_pred = get_predictions(model, test_ds)

    report = classification_report(y_true, y_pred, digits=4)
    print(report)

def length_bucket_eval(test_data_raw, tokenizer, model, max_length=32):
    print("Evaluating by text length buckets...")

    def bucket(length):
        if length <= 5:
            return "short"
        elif length <= 12:
            return "medium"
        else:
            return "long"

    test_data_raw = test_data_raw.map(lambda x: {"length": len(x["text"].split()), "bucket": bucket(len(x["text"].split()))})

    for bucket_name in ["short", "medium", "long"]:
        subset = test_data_raw.filter(lambda x: x["bucket"] == bucket_name)
        subset = subset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=max_length), batched=True)
        tf_subset = subset.to_tf_dataset(columns=["input_ids", "attention_mask"], label_cols="intent", batch_size=32)

        y_true, y_pred = get_predictions(model, tf_subset)
        print(f"\n {bucket_name.upper()} ({len(y_true)} samples):")
        print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":

    model = TFAutoModelForSequenceClassification.from_pretrained("./intent_model_tf")
    tokenizer = AutoTokenizer.from_pretrained("./intent_model_tf")

    evaluate_classification_metrics()

    test_raw = load_dataset("clinc_oos", "plus")["test"] \
                 .filter(lambda x: x["intent"] != 42)

    length_bucket_eval(test_raw, tokenizer, model)
