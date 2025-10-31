# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:57:37 2025

@author: THYAGHARAJAN
"""

import matplotlib.pyplot as plt  # For displaying the images
import tensorflow as tf
import math    #used ceil round up


# In[]:
def display_augmented_images(Keras_Dataset, no_of_batches, class_names, no_of_images_per_row):
    """Displays augmented images in a grid format.

    Args:
        Keras_Dataset: The Keras dataset containing augmented images.
        no_of_batches: The number of batches to randomly sample from.
        class_names: A list of class names corresponding to the labels.
        no_of_images_per_row: The number of images to display in each row of the grid.
    """
    batch_no = 1  # batch number
    cols = no_of_images_per_row
    # Create a figure with appropriate size for the grid
    plt.figure(figsize=(cols * 2, no_of_batches * 2))  # avoiding *2 will create  overlapped image display
    
    for image_batch, label_batch in Keras_Dataset.take(no_of_batches):
        print(f"\nImage Batch shape: {image_batch.shape}, dtype: {image_batch.dtype}, Label: {label_batch.numpy()}")

        if batch_no <= no_of_batches:
            for i in range(no_of_images_per_row):  # j is the column index for the figure
                # Select image and convert to uint8 format
                image = image_batch[i].numpy().astype("uint8")
                # Create a subplot for the image
                plt.subplot(no_of_batches, cols, (batch_no - 1) * no_of_images_per_row + i + 1)
                plt.imshow(image)
            
                # Display class label and turn off axis ticks
                class_label = class_names[label_batch[i].numpy()]
                plt.title(f"Class Label: {class_label}")
                plt.axis("off")

        batch_no += 1  # Increment the batch number

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()

# In[]:


def predict_display_images1(KKT_model, dataset, target_subdirs):
    # Assuming your target_subdirs contains the class names
    """Shuffle the dataset, predict a random batch, and display images with predicted labels."""
    # Shuffle the dataset and return a random batch
    buffer_size=1000 #
    random_batch = dataset.shuffle(buffer_size).take(1)  # Shuffle and take one batch
    images, labels = next(iter(random_batch))  # Get images and labels from the batch

    # Use the model to predict the labels for the batch
    predictions = KKT_model.predict(images)  # Shape: (batch_size, num_classes)
    predicted_labels = tf.argmax(predictions, axis=1)  # Get predicted class labels
    
    # Map numerical labels to class names using the target_subdirs list
    predicted_class_names = [target_subdirs[label] for label in predicted_labels.numpy()]
    true_class_names = [target_subdirs[label] for label in labels.numpy()]

    # Plot the images with predicted and true labels
    batch_size = images.shape[0]
    plt.figure(figsize=(10, 10))

    for i in range(batch_size):
        plt.subplot(3, math.ceil(batch_size/3), i+1)  # Adjust the grid size depending on batch size
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Pred: {predicted_class_names[i]}\nTrue: {true_class_names[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# In[]
def predict_display_images(KKT_model, dataset, class_names_list):
    # Parameter is now named class_names_list for clarity
    """Shuffle the dataset, predict a random batch, and display images with predicted labels."""
    
    # Shuffle the dataset and return a random batch
    buffer_size=1000 
    random_batch = dataset.shuffle(buffer_size).take(1)  # Shuffle and take one batch
    images, labels = next(iter(random_batch))  # Get images and labels from the batch

    # Use the model to predict the labels for the batch
    predictions = KKT_model.predict(images)  # Shape: (batch_size, num_classes)
    predicted_labels = tf.argmax(predictions, axis=1)  # Get predicted class labels
    
    # Map numerical labels to class names using the class_names_list
    # Note: The mapping logic remains the same, just the variable name changed.
    predicted_class_names = [class_names_list[label] for label in predicted_labels.numpy()]
    true_class_names = [class_names_list[label] for label in labels.numpy()]

    # Plot the images with predicted and true labels
    batch_size = images.shape[0]
    plt.figure(figsize=(10, 10))

    for i in range(batch_size):
        # Adjust the grid size depending on batch size (using a 3-row layout)
        plt.subplot(3, math.ceil(batch_size/3), i+1) 
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Pred: {predicted_class_names[i]}\nTrue: {true_class_names[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()







