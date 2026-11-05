import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 1: Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Step 2: Normalize pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Step 3: Add the channel dimension
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

# Step 4: Build the CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Step 5: Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 6: Display the model architecture
model.summary()

# Step 7: Train the model
model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.1
)

# Step 8: Evaluate the model
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 9: Save the trained model
os.makedirs("model", exist_ok=True)

model.save("model/handwritten_digit_model.keras")

print("Model saved successfully!")