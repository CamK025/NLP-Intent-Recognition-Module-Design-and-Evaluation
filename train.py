# train.py

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, create_optimizer
import tensorflow as tf
from data_preprocessing import load_and_preprocess_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model_custom_loop(
    model_name="bert-base-uncased",
    epochs=10,
    learning_rate=5e-5,
    batch_size=32,
    max_length=32,
    intent_limit=None,
    save_path="./intent_model_tf",
    early_stopping=True,
    patience=5
):
    print("Loading data...")
    train_ds, val_ds, test_ds, num_labels, tokenizer = load_and_preprocess_dataset(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        intent_limit=intent_limit
    )

    print(f"Using {num_labels} intent classes.")
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    num_train_steps = tf.data.experimental.cardinality(train_ds).numpy() * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    optimizer, schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    epoch_loss_history = []
    val_loss_history = []
    val_acc_history = []
    best_val_acc = 0.0
    no_improve_count = 0

    print("Starting training ...")
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        train_acc.reset_states()
        val_acc.reset_states()
        epoch_loss = tf.keras.metrics.Mean()
        val_epoch_loss = tf.keras.metrics.Mean()

        for batch in tqdm(train_ds, desc="Training"):
            features, labels = batch
            x = {k: features[k] for k in ["input_ids", "attention_mask"]}
            y = labels

            with tf.GradientTape() as tape:
                logits = model(x, training=True).logits
                loss = loss_fn(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_acc.update_state(y, logits)
            epoch_loss.update_state(loss)

        avg_loss = epoch_loss.result().numpy()
        epoch_loss_history.append(avg_loss)

        for batch in val_ds:
            features, labels = batch
            x_val = {k: features[k] for k in ["input_ids", "attention_mask"]}
            y_val = labels
            val_logits = model(x_val, training=False).logits
            loss_val = loss_fn(y_val, val_logits)
            val_acc.update_state(y_val, val_logits)
            val_epoch_loss.update_state(loss_val)

        val_loss_val = val_epoch_loss.result().numpy()
        val_acc_val = val_acc.result().numpy()
        val_acc_history.append(val_acc_val)
        val_loss_history.append(val_loss_val)

        print(f"Training Acc: {train_acc.result().numpy():.4f} | Train Loss: {avg_loss:.4f}")
        print(f"Validation Acc: {val_acc_val:.4f} | Val Loss: {val_loss_val:.4f}")

        # Early stopping logic
        if early_stopping:
            if val_acc_val > best_val_acc:
                best_val_acc = val_acc_val
                no_improve_count = 0
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"New best model saved with val acc: {val_acc_val:.4f}")
            else:
                no_improve_count += 1
                print(f"No improvement for {no_improve_count} epoch(s).")

            if no_improve_count >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Plotting loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_loss_history)+1), epoch_loss_history, marker="o", label="Training Loss")
    plt.plot(range(1, len(val_loss_history)+1), val_loss_history, marker="s", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Over Epochs")
    plt.legend()
    plt.savefig("training_vs_val_loss.png")
    # plt.show()


if __name__ == "__main__":
    train_model_custom_loop()
