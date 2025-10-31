# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:19:45 2025

@author: THYAGHARAJAN
"""
# In[]:
import tensorflow as tf
import os
from glob import glob
import streamlit as st
import matplotlib.pyplot as plt

# In[]
def MultiClass_Prediction(KKT_model, IMG_SIZE, test_folder, class_names):
    # Function to load and preprocess a single image
    def load_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.expand_dims(image, axis=0)  # Add batch dimension for prediction
        return image

    # Function to predict and display results
    def predict_and_display(model, image_paths, class_names):
        for image_path in image_paths:
            # Load and preprocess the image
            image = load_image(image_path)

            # Get predictions (logits)
            logits = model.predict(image)

            # Apply softmax to get probabilities
            probabilities = tf.nn.softmax(logits).numpy().squeeze()

            # Determine the predicted class
            predicted_class_index = tf.argmax(probabilities).numpy()
            predicted_class = class_names[predicted_class_index]
            confidence = probabilities[predicted_class_index]

            # Display the image with prediction
            plt.figure(figsize=(4, 4))
            img = tf.image.decode_image(tf.io.read_file(image_path), channels=3).numpy()
            plt.imshow(img.astype("uint8"))
            plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
            plt.axis("off")
            plt.show()

    # Collect all image paths
    image_paths = glob(os.path.join(test_folder, "*.jpg")) + glob(os.path.join(test_folder, "*.png"))

    # Ensure there are images to predict
    if not image_paths:
        print("No images found in the test folder.")
        return

    # Call the inner function
    predict_and_display(KKT_model, image_paths, class_names)  # This is not the main call

# In[]

def MultiClass_Prediction_Streamlit(KKT_model, IMG_SIZE, test_folder, class_names):
    # Function to load and preprocess a single image
    def load_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.expand_dims(image, axis=0)  # Add batch dimension for prediction
        return image

    # Function to predict and display results
    def predict_and_display(model, image_paths, class_names):
        for image_path in image_paths:
            # Load and preprocess the image
            image = load_image(image_path)

            # Get predictions (logits)
            logits = model.predict(image)

            # Apply softmax to get probabilities
            probabilities = tf.nn.softmax(logits).numpy().squeeze()

            # Determine the predicted class
            predicted_class_index = tf.argmax(probabilities).numpy()
            predicted_class = class_names[predicted_class_index]
            confidence = probabilities[predicted_class_index]

            # Display the image with prediction using Streamlit compatible commands
            
            # Create the Matplotlib figure (Same as before)
            fig, ax = plt.subplots(figsize=(4, 4)) 
            
            #Decode and display the image onto the figure (Same as before)
            img = tf.image.decode_image(tf.io.read_file(image_path), channels=3).numpy()
            ax.imshow(img.astype("uint8"))
            ax.set_title(f"Predicted: {predicted_class} ({confidence:.2f})")
            ax.axis("off")
            
            # Use Streamlit to render the Matplotlib figure
            st.pyplot(fig) # <--- THIS IS THE FIX
    
            # Display the text results separately for clarity
            st.markdown(f"**{os.path.basename(image_path)}:** Predicted **{predicted_class}** with **{confidence:.2f}** confidence.")
            
            # Optional: Add a separator
            st.markdown("---")

    # Collect all image paths
    image_paths = glob(os.path.join(test_folder, "*.jpg")) + glob(os.path.join(test_folder, "*.png"))

    # Ensure there are images to predict
    if not image_paths:
        print("No images found in the test folder.")
        return

    # Call the inner function
    predict_and_display(KKT_model, image_paths, class_names)  # This is not the main call

# In[]:
#Check this function parameters
import tensorflow as tf 
def Multi_class_prediction_All_Probabilities(model, IMG_SIZE, image_paths, class_names):
    def load_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)  # Use decode_jpeg if all images are .jpg
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.expand_dims(image, axis=0)  # Add batch dimension for prediction
        return image
    for image_path in image_paths:
        # Load and preprocess the image
        image = load_image(image_path)

        # Get predictions (logits)
        logits = model.predict(image)

        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(logits).numpy().squeeze()

        # Display the image with prediction
        plt.figure(figsize=(4, 4))
        img = tf.image.decode_image(tf.io.read_file(image_path), channels=3).numpy()
        plt.imshow(img.astype("uint8"))
        plt.title("\n".join([f"{name}: {prob:.2f}" for name, prob in zip(class_names, probabilities)]))
        plt.axis("off")
        plt.show()

# In[]:
import tensorflow as tf
import os
from glob import glob
import matplotlib.pyplot as plt

def Binary_Class_Prediction(KKT_model, IMG_SIZE, test_folder, class_names):
    # Function to load and preprocess a single image
    def load_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)  # Use decode_jpeg if all images are .jpg
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.expand_dims(image, axis=0)  # Add batch dimension for prediction
        return image

    # Function to predict and display results
    def predict_and_display(model, image_paths, class_names):
        for image_path in image_paths:
            # Load and preprocess the image
            image = load_image(image_path)

            # Get predictions (logits)
            logits = model.predict(image)

            # Apply sigmoid to get probabilities
            prediction = tf.nn.sigmoid(logits).numpy().squeeze()  # Apply sigmoid to logits

            # Determine the predicted class based on the probability threshold of 0.5
            predicted_class = class_names[1] if prediction >= 0.5 else class_names[0]
            confidence = prediction if prediction >= 0.5 else 1 - prediction

            # Display the image with prediction
            plt.figure(figsize=(4, 4))
            img = tf.image.decode_image(tf.io.read_file(image_path), channels=3).numpy()
            plt.imshow(img.astype("uint8"))
            plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
            plt.axis("off")
            plt.show()

    # Collect all image paths
    image_paths = glob(os.path.join(test_folder, "*.jpg")) + glob(os.path.join(test_folder, "*.png"))

    # Ensure there are images to predict
    if not image_paths:
        print("No images found in the test folder.")
        return

    # Call the inner function
    predict_and_display(KKT_model, image_paths, class_names)
    
# In[]:
