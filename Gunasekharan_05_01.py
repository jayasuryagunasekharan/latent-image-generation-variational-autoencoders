# Gunasekharan, Jayasurya
# 1002_060_437
# 2023_12_03
# Assignment_05_01


# Import necessary libraries
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib.widgets import Slider
from tensorflow.keras import layers, models


# Function to fetch the LFW dataset
def fetch_lfw_dataset(resize_factor=0.4):
    # Load LFW dataset
    lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=resize_factor)

    # Extract images and attributes
    lfw_images = lfw_dataset.images
    lfw_attributes = lfw_dataset.target

    return lfw_images, lfw_attributes


# Load the LFW dataset using the fetch_lfw_dataset function
data, attrs = fetch_lfw_dataset()

# Display a random selection of 20 images from the LFW dataset
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    idx = random.randint(0, 20)
    img = data[idx]
    plt.imshow(img, cmap="gray")
    # plt.title(f"Person ID: {attrs[idx]}")  # Commented out to match the masked images
    plt.axis("off")

    # Save only the first image
    if i == 0:
        plt.savefig("celeb_image.png", bbox_inches="tight", pad_inches=0.1)

plt.show()


data = np.array(data / 255, dtype="float32")
X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)


class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = models.Sequential(
            [
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(latent_dim + latent_dim),
            ]
        )

        # Decoder
        self.decoder = models.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(512, activation="relu"),
                layers.Dense(50 * 37, activation="sigmoid"),
                layers.Reshape((50, 37, 1)),
            ]
        )

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mean, logvar


def vae_loss(x, x_reconstructed, mean, logvar):
    reconstruction_loss = tf.keras.losses.binary_crossentropy(x, x_reconstructed)
    reconstruction_loss *= 50 * 37  # Adjust based on image dimensions

    kl_loss = 1 + logvar - tf.square(mean) - tf.exp(logvar)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    return total_loss


# Function to generate and save images during training
def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.decode(z)

    # Limit the number of displayed images to 16
    num_images = min(predictions.shape[0], 16)

    fig = plt.figure(figsize=(4, 4))

    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.show()


# Instantiate the VAE model
latent_dim = 6  # Adjust as needed
vae_model = VAE(latent_dim)

# Compile the model with an optimizer (e.g., Adam) and no loss function
vae_model.compile(optimizer="adam", loss=vae_loss)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


# Function to compute the loss
def compute_loss(model, x):
    x_reconstructed, mean, logvar = model(x)

    # Reshape x to match the shape of x_reconstructed
    x_reshaped = tf.reshape(x, [-1, 50, 37, 1])

    reconstruction_loss = tf.keras.losses.binary_crossentropy(
        x_reshaped, x_reconstructed
    )
    reconstruction_loss *= 50 * 37  # Adjust based on image dimensions

    kl_loss = 1 + logvar - tf.square(mean) - tf.exp(logvar)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    return tf.reduce_mean(reconstruction_loss) + tf.reduce_mean(kl_loss)


# Function to perform a training step
@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Training loop
epochs = 10  # Adjust as needed
batch_size = 32  # Adjust as needed

for epoch in range(epochs):
    for batch in range(0, len(X_train), batch_size):
        x_batch = X_train[batch : batch + batch_size]
        loss = train_step(vae_model, x_batch, optimizer)

    # Display progress
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

    # Generate and save images after each epoch
    # generate_and_save_images(vae_model, epoch + 1, X_val)

# Save the VAE model
vae_model.save_weights("vae_lfw_model.h5")


class Masked_VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Masked_VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = models.Sequential(
            [
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(latent_dim + latent_dim),
            ]
        )

        # Decoder
        self.decoder = models.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(512, activation="relu"),
                layers.Dense(50 * 37, activation="sigmoid"),
                layers.Reshape((50, 37, 1)),
            ]
        )

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs):
        x, mask = inputs
        # print("x shape: ", x.shape)
        # print("mask shape: ", mask.shape)

        # Element-wise multiplication
        x_masked = x * mask
        # print("x_masked shape: ", x_masked.shape)

        mean, logvar = self.encode(x_masked)
        z = self.reparameterize(mean, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mean, logvar


def vae_loss(x, x_reconstructed, mean, logvar):
    reconstruction_loss = tf.keras.losses.binary_crossentropy(x, x_reconstructed)
    reconstruction_loss *= 50 * 37  # Adjust based on image dimensions

    kl_loss = 1 + logvar - tf.square(mean) - tf.exp(logvar)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    return total_loss


# Step 7: Modify the Model for Masked Images
def reparameterize(args):
    mean, logvar = args
    eps = tf.random.normal(shape=tf.shape(mean))
    return eps * tf.exp(logvar * 0.5) + mean


# Step 8: Load and Preprocess Masked Images
# Assuming you have a function to load masked images, replace the following line accordingly
# masked_data, masked_attrs = fetch_masked_lfw_dataset()
def fetch_masked_lfw_dataset(num_images=64, resize_factor=0.4):
    # Add mask to lfw_dataset images and attributes
    masked_images = []
    masked_attributes = []

    lfw_dataset = fetch_lfw_people(min_faces_per_person=35, resize=resize_factor)

    for i in range(num_images - 3):
        img = lfw_dataset.images[i]
        masked_img = np.copy(img)
        masked_img[20:35, 15:35] = 0.0
        masked_images.append(masked_img)

        # Convert the integer attributes to strings
        masked_attributes.append(str(lfw_dataset.target[i]) + "_mask")

    return np.array(masked_images), np.array(masked_attributes)


# Usage example
masked_data, masked_attrs = fetch_masked_lfw_dataset()
# print("masked_data", masked_data.shape)

plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    idx = random.randint(0, len(masked_data) - 1)
    plt.imshow(masked_data[idx], cmap="gray")
    plt.axis("off")

plt.show()


# Normalize the masked data
masked_data = np.array(masked_data / 255, dtype="float32")

# Split the Masked Data into Train and Test Sets
X_masked_train, X_masked_val = train_test_split(
    masked_data, test_size=0.2, random_state=42
)

# Step 9: Train the Model
vae_masked_model = Masked_VAE(latent_dim)


# Compile the model with an optimizer and loss function
vae_masked_model.compile(optimizer="adam", loss=vae_loss)


# Define the optimizer
optimizer_masked = tf.keras.optimizers.Adam(learning_rate=1e-3)


# Function to compute the loss for masked images
# @tf.function
def compute_masked_loss(model, x, mask):
    x_reconstructed, mean, logvar = model([x, mask])

    # Reshape x to match the shape of x_reconstructed
    x_reshaped = tf.reshape(x, [-1, 50, 37, 1])

    reconstruction_loss = tf.keras.losses.binary_crossentropy(
        x_reshaped, x_reconstructed
    )
    reconstruction_loss *= 50 * 37  # Adjust based on image dimensions

    kl_loss = 1 + logvar - tf.square(mean) - tf.exp(logvar)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    # Compute mean of reconstruction and KL losses
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    kl_loss = tf.reduce_mean(kl_loss)

    return reconstruction_loss, kl_loss


# Function to perform a training step for masked images
# @tf.function
def train_masked_step(model, x, mask, optimizer):
    with tf.GradientTape() as tape:
        reconstruction_loss, kl_loss = compute_masked_loss(model, x, mask)
        loss = reconstruction_loss + kl_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, reconstruction_loss, kl_loss


# Training loop for masked images
epochs_masked = 10  # Adjust as needed
batch_size_masked = 16  # Adjust as needed

for epoch in range(epochs_masked):
    for batch in range(0, len(X_masked_train), batch_size_masked):
        x_batch = X_masked_train[batch : batch + batch_size_masked]
        mask_batch = masked_data[batch : batch + batch_size_masked]
        loss, reconstruction_loss, kl_loss = train_masked_step(
            vae_masked_model, x_batch, mask_batch, optimizer_masked
        )

    print(
        f"Epoch {epoch + 1}/{epochs_masked}, Loss: {loss.numpy()}, "
        f"Reconstruction Loss: {reconstruction_loss.numpy()}, "
        f"KL Loss: {kl_loss.numpy()}"
    )

# Save the VAE model
vae_masked_model.save_weights("vae_lfw_masked_model.h5")

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


original_image = None
label_reconstructed_vae1 = None
label_reconstructed_vae2 = None


# Function to update images based on slider values (Task 1)
def update_images(*args):
    latent_values = [slider_var[i].get() for i in range(latent_dim)]
    latent_vector = np.array(latent_values).reshape(1, latent_dim)
    reconstructed_image = vae_model.decode(latent_vector)
    reconstructed_image = reconstructed_image.numpy().reshape((50, 37))

    ax.clear()
    ax.imshow(reconstructed_image, cmap="gray")
    canvas_task1.draw()


# Function to apply a square mask to an image (Task 2)
def apply_mask(image, mask_size, top_left_x, top_left_y):
    h, w = image.shape[:2]
    mask = np.ones_like(image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask_top_left_x = int(top_left_x * w)
    mask_top_left_y = int(top_left_y * h)
    mask_size = int(mask_size * min(h, w))

    mask[
        mask_top_left_y : mask_top_left_y + mask_size,
        mask_top_left_x : mask_top_left_x + mask_size,
    ] = 0

    masked_image = image * mask
    return masked_image


def custom_mean_squared_error(image1, image2):
    # print("image1 , image2", image1.shape, image2.shape)
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Resize the larger array to match the size of the smaller one
    if h1 * w1 > h2 * w2:
        image1 = cv2.resize(image1, (w2, h2))
    elif h1 * w1 < h2 * w2:
        image2 = cv2.resize(image2, (w1, h1))

    mse = np.mean((image1 - image2) ** 2)
    return mse


# Function to update image when sliders change (Task 2)
def on_change_mask(*args):
    update_image()


# Function to update image when sliders change (Task 2)
def on_change_position(*args):
    update_image()


# Function to update image based on sliders
def update_image():
    global original_image, label_reconstructed_vae1, label_reconstructed_vae2, canvas  # Add canvas to global variables

    # Check if the original image is defined
    if original_image is None:
        return

    # Get the slider values
    mask_width = mask_width_slider.get()
    top_left_x = top_left_x_slider.get()
    top_left_y = top_left_y_slider.get()

    # Apply the mask to the loaded image
    masked_image = apply_mask(original_image, mask_width, top_left_x, top_left_y)

    # Display the reconstructed image with the mask using VAE 1
    latent_values = [slider_var[i].get() for i in range(latent_dim)]
    latent_vector = np.array(latent_values).reshape(1, latent_dim)
    # Decode latent vector using VAE 1
    reconstructed_image_vae1 = vae_model.decode(latent_vector).numpy()
    # Reshape to (50, 37)
    reconstructed_image_vae1_reshaped = reconstructed_image_vae1.reshape((50, 37))

    # Decode latent vector using VAE 2
    reconstructed_image_vae2 = vae_masked_model.decode(latent_vector).numpy()
    # Reshape to (50, 37)
    reconstructed_image_vae2_reshaped = reconstructed_image_vae2.reshape((50, 37))

    # Clear the previous plots
    plt.clf()

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # Display the reconstructed images
    ax[0].imshow(reconstructed_image_vae1_reshaped, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("VAE 1 Reconstructed")

    ax[1].imshow(reconstructed_image_vae2_reshaped, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("VAE 2 Reconstructed")

    # Attach the plot to Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=4, padx=1, pady=1)

    # Calculate and display the Mean Squared Error for VAE 1
    mse_vae1 = custom_mean_squared_error(
        original_image,
        reconstructed_image_vae1_reshaped,
    )
    mse_label_vae1.config(text=f"MSE (VAE 1): {mse_vae1:.4f}")

    # Calculate and display the Mean Squared Error for VAE 2
    mse_vae2 = custom_mean_squared_error(
        masked_image,
        reconstructed_image_vae2_reshaped,
    )
    mse_label_vae2.config(text=f"MSE (VAE 2): {mse_vae2:.4f}")

    # Configure or create label_reconstructed_vae1
    if label_reconstructed_vae1 is None:
        label_reconstructed_vae1 = ttk.Label(frame_task2)
        label_reconstructed_vae1.grid(row=0, column=5, padx=1, pady=1)

    label_reconstructed_vae1.configure(
        image=ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(reconstructed_image_vae1_reshaped * 255), "L"
            )
        )
    )
    label_reconstructed_vae1.image = ImageTk.PhotoImage(
        image=Image.fromarray(np.uint8(reconstructed_image_vae1_reshaped * 255), "L")
    )

    # Configure or create label_reconstructed_vae2
    if label_reconstructed_vae2 is None:
        label_reconstructed_vae2 = ttk.Label(frame_task2)
        label_reconstructed_vae2.grid(row=0, column=6, padx=1, pady=1)

    # Reshape or squeeze reconstructed_image_vae2 if it has more than 2 dimensions
    reconstructed_image_vae2_reshaped = np.squeeze(reconstructed_image_vae2)

    label_reconstructed_vae2.configure(
        image=ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(reconstructed_image_vae2_reshaped * 255), "L"
            )
        )
    )
    label_reconstructed_vae2.image = ImageTk.PhotoImage(
        image=Image.fromarray(np.uint8(reconstructed_image_vae2_reshaped * 255), "L")
    )

    # Draw the canvas for Task 2
    canvas.draw()


# Function to upload an image (Task 2)
def upload_image():
    global original_image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
    )

    if file_path:
        uploaded_image = Image.open(file_path)
        uploaded_image = uploaded_image.convert("L")
        uploaded_image = uploaded_image.resize((300, 300))
        original_image = np.array(uploaded_image) / 255.0

        update_image()


# Create Tkinter window
root = tk.Tk()
root.title("VAE")

# Task 1 GUI (top part)
frame_task1 = ttk.Frame(root)
frame_task1.grid(row=0, column=0, pady=10, padx=10)

# Add heading label for Task 1
label_heading_task1 = ttk.Label(frame_task1, text="Task 1", font=("Helvetica", 16))
label_heading_task1.grid(row=0, column=1, pady=10)

# Sliders for Task 1
slider_var = [tk.DoubleVar() for _ in range(latent_dim)]
sliders = []

label_latent_variables = ttk.Label(frame_task1, text="Latent Variables")
label_latent_variables.grid(row=1, column=1, pady=5)

for i in range(latent_dim):
    slider = ttk.Scale(
        frame_task1, from_=-3, to=3, variable=slider_var[i], command=update_images
    )
    slider.grid(row=i + 2, column=0, pady=5, padx=10)
    sliders.append(slider)

fig_task1 = plt.Figure(figsize=(4, 4), dpi=100)
ax = fig_task1.add_subplot(111)
canvas_task1 = FigureCanvasTkAgg(fig_task1, master=frame_task1)
canvas_widget_task1 = canvas_task1.get_tk_widget()
canvas_widget_task1.grid(row=2, column=1, rowspan=3, padx=10, pady=5)

# Set initial slider values for Task 1
initial_values_task1 = [0.0] * latent_dim
for i in range(latent_dim):
    slider_var[i].set(initial_values_task1[i])

# Initialize images based on initial slider values for Task 1
update_images()

# Task 2 GUI (bottom part)
frame_task2 = ttk.Frame(root)
frame_task2.grid(row=1, column=0, pady=10, padx=10)

# Add heading label for Task 2
label_heading_task2 = ttk.Label(frame_task2, text="Task 2", font=("Helvetica", 16))
label_heading_task2.grid(row=0, column=1, pady=10)

# Sliders and controls for Task 2
mask_width_slider = ttk.Scale(
    frame_task2,
    from_=0,
    to=0.5,
    orient=tk.HORIZONTAL,
    cursor="circle",
    command=on_change_mask,
)
top_left_x_slider = ttk.Scale(
    frame_task2,
    from_=0,
    to=1,
    orient=tk.HORIZONTAL,
    cursor="circle",
    command=on_change_position,
)
top_left_y_slider = ttk.Scale(
    frame_task2,
    from_=0,
    to=1,
    orient=tk.VERTICAL,
    cursor="circle",
    command=on_change_position,
)

initial_mask_width = 0.5 * (mask_width_slider["from"] + mask_width_slider["to"])
initial_top_left_x = 0.5 * (top_left_x_slider["from"] + top_left_x_slider["to"])
initial_top_left_y = 0.5 * (top_left_y_slider["from"] + top_left_y_slider["to"])

mask_width_slider.set(initial_mask_width)
top_left_x_slider.set(initial_top_left_x)
top_left_y_slider.set(initial_top_left_y)

label_mask_width = ttk.Label(frame_task2, text="Mask Width:")
label_top_left_x = ttk.Label(frame_task2, text="Top Left X:")
label_top_left_y = ttk.Label(frame_task2, text="Top Left Y:")

label_mask_width_start = ttk.Label(frame_task2, text=f"{mask_width_slider['from']:.2f}")
label_mask_width_end = ttk.Label(frame_task2, text=f"{mask_width_slider['to']:.2f}")
label_top_left_x_start = ttk.Label(frame_task2, text=f"{top_left_x_slider['from']:.2f}")
label_top_left_x_end = ttk.Label(frame_task2, text=f"{top_left_x_slider['to']:.2f}")
label_top_left_y_start = ttk.Label(frame_task2, text=f"{top_left_y_slider['from']:.2f}")
label_top_left_y_end = ttk.Label(frame_task2, text=f"{top_left_y_slider['to']:.2f}")

label_uploaded = ttk.Label(frame_task2)
label_uploaded.grid(row=1, column=1, rowspan=3, padx=10, pady=5)

label_mask_width.grid(row=1, column=0, padx=10, pady=5)
mask_width_slider.grid(row=1, column=2, padx=10, pady=5)
label_mask_width_start.grid(row=1, column=1, padx=5, pady=5)
label_mask_width_end.grid(row=1, column=3, padx=5, pady=5)

label_top_left_x.grid(row=2, column=0, padx=10, pady=5)
top_left_x_slider.grid(row=2, column=2, padx=10, pady=5)
label_top_left_x_start.grid(row=2, column=1, padx=5, pady=5)
label_top_left_x_end.grid(row=2, column=3, padx=5, pady=5)

label_top_left_y.grid(row=3, column=0, padx=10, pady=5)
top_left_y_slider.grid(row=3, column=2, padx=10, pady=5, rowspan=2, sticky="ns")
label_top_left_y_start.grid(row=3, column=1, padx=5, pady=5)
label_top_left_y_end.grid(row=3, column=3, padx=5, pady=5)

upload_button = ttk.Button(frame_task2, text="Load New Image", command=upload_image)
upload_button.grid(row=4, column=3, columnspan=2, pady=10)

mse_label_vae1 = ttk.Label(frame_task2, text="MSE (VAE 1):")
mse_label_vae1.grid(row=1, column=9, padx=10, pady=5)

mse_label_vae2 = ttk.Label(frame_task2, text="MSE (VAE 2):")
mse_label_vae2.grid(row=1, column=10, padx=10, pady=5)

# Run the Tkinter event loop
root.mainloop()
