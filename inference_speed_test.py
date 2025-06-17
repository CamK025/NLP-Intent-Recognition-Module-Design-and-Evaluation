# inference_speed_test.py

import time
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

def measure_latency(model, tokenizer, text="how do I cancel my order", repeat=100):
    print(f"Measuring inference latency over {repeat} runs...")
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)

    model(inputs)

    start = time.time()
    for _ in range(repeat):
        model(inputs)
    end = time.time()

    avg_time_ms = (end - start) * 1000 / repeat
    print(f"Average inference time: {avg_time_ms:.2f} ms/sample")

if __name__ == "__main__":
    model = TFAutoModelForSequenceClassification.from_pretrained("./intent_model_tf")
    tokenizer = AutoTokenizer.from_pretrained("./intent_model_tf")
    measure_latency(model, tokenizer)
