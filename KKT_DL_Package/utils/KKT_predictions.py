# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:19:45 2025

@author: THYAGHARAJAN
"""
# In[]:
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from glob import glob
import streamlit as st
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import math
from huggingface_hub import list_repo_files, hf_hub_download
import shutil

# In[]

def multiclass_prediction_return(model_full_path, test_folder_path, class_names,IMG_SIZE= (224, 224)):
    '''
    Don't chnage the name of this function
    Predicts the labels of the images in test_folder and returns

    Parameters
    ----------
    KKT_model : .keras  modelname will be loaded from the given path

    test_folder : model and .txt file that contains the list of class names are in the same folder 
    class_names : list of the class names, don't give the path of the .txt file'

    Returns
    -------
    all_image_paths, all_predicted_labels, all_confidences
    All are list

    '''
    
    # -------------------------------------------------------------------------
    # Helper Functions (Keep as they are fine for prediction)
    # -------------------------------------------------------------------------
    def load_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        # Scale to 0-1 for common models, assuming KKT_model expects this
        image = image / 255.0 
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        return image

    # -------------------------------------------------------------------------
    # Prediction and Data Collection (Modified)
    # -------------------------------------------------------------------------
    
    # Initialization of lists to return
    all_image_paths = []
    all_predicted_labels = []
    all_confidences = [] # Storing confidence (max probability)

    # Collect all image paths
    image_paths_to_process = glob(os.path.join(test_folder_path, "*.jpg")) + \
                             glob(os.path.join(test_folder_path, "*.png"))
    
    # Ensure there are images to predict
    if not image_paths_to_process:
        print("No images found in the test folder.")
        # Return empty lists if no images are found
        return [], [], []

    print(f"Starting predictions for {len(image_paths_to_process)} images...")
    print("\n🧠 Loading model...")
    KKT_model = load_model(model_full_path)
    
    for image_path in image_paths_to_process:
        
        # Load and preprocess the image
        image = load_image(image_path)

        # Get predictions (logits)
        # NOTE: Using KKT_model.predict(image) is standard for Keras.
        logits = KKT_model.predict(image, verbose=0) 

        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(logits).numpy().squeeze()

        # Determine the predicted class
        predicted_class_index = tf.argmax(probabilities).numpy()
        predicted_class = class_names[predicted_class_index]
        confidence = probabilities[predicted_class_index]

        # -----------------------------------------------------------------
        # 💾 COLLECT DATA
        # -----------------------------------------------------------------
        all_image_paths.append(image_path)
        all_predicted_labels.append(predicted_class)
        all_confidences.append(confidence)
        

    # -----------------------------------------------------------------
    # 🚀 RETURN DATA
    # -----------------------------------------------------------------
    # Return the collected data: paths, labels, and confidence values
    return all_image_paths, all_predicted_labels, all_confidences

# In[]

def display_images_gui(all_image_paths,  all_labels, img_size=(224, 224)):
    """
    Don't chnage the name of this function
    Display predicted images with labels in a paged Tkinter popup GUI.
    Includes grid size selector, page navigation, and adaptive font scaling.
    """
    #breakpoint()
    if not all_image_paths or not all_labels:
        print("⚠️ No images or predictions to display.")
        return

    total_images = len(all_image_paths)

    # --- Tkinter window setup ---
    root = tk.Tk()
    root.title("🦋 Insect Predictions Viewer")
    root.configure(bg="#1e1e1e")

    # --- Top control frame ---
    control_frame = tk.Frame(root, bg="#1e1e1e")
    control_frame.pack(pady=8)

    tk.Label(control_frame, text="Grid size:", fg="white", bg="#1e1e1e", font=("Segoe UI", 10)).pack(side="left", padx=5)

    grid_var = tk.StringVar(value="2x3")
    grid_menu = ttk.Combobox(
        control_frame,
        textvariable=grid_var,
        values=["1x1", "2x2", "2x3", "3x3", "4x4"],
        width=5,
        state="readonly"
    )
    grid_menu.pack(side="left", padx=5)

    # --- Frame for image grid ---
    frame_images = tk.Frame(root, bg="#1e1e1e")
    frame_images.pack(padx=10, pady=10, fill="both", expand=True)

    # --- Pagination variables ---
    current_page = tk.IntVar(value=0)
    total_pages = tk.IntVar(value=1)

    # --- Function to choose font size based on grid density ---
    def get_font_size(grid_str):
        rows, cols = map(int, grid_str.split("x"))
        density = rows * cols
        if density <= 2:
            return 11
        elif density <= 6:
            return 10
        elif density <= 9:
            return 9
        elif density <= 12:
            return 8
        else:
            return 7  # For 4x4 and higher
    # --- Draw grid function ---
    def draw_grid():
        """Display current page of predictions based on grid size."""
        for widget in frame_images.winfo_children():
            widget.destroy()

        grid_choice = grid_var.get()
        rows, cols = map(int, grid_choice.split("x"))
        per_page = rows * cols
        total_pages.set(math.ceil(total_images / per_page))

        font_size = get_font_size(grid_choice)
        page = current_page.get()
        start_idx = page * per_page
        end_idx = min(start_idx + per_page, total_images)

        for i, (img_path, label_text) in enumerate(zip(all_image_paths[start_idx:end_idx],
                                                       all_labels[start_idx:end_idx])):
            try:
                img = Image.open(img_path)
                img = img.resize(img_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)

                # Image widget
                lbl_img = tk.Label(frame_images, image=photo, bg="#1e1e1e")
                lbl_img.image = photo
                lbl_img.grid(row=(i // cols) * 2, column=i % cols, padx=10, pady=5)

                # Label widget
                lbl_text = tk.Label(
                    frame_images,
                    text=label_text,
                    bg="#2b2b2b",
                    fg="lime",
                    font=("Segoe UI", font_size, "bold"),
                    wraplength=img_size[0],
                    justify="center"
                )
                lbl_text.grid(row=(i // cols) * 2 + 1, column=i % cols, padx=10, pady=(0, 10))
            except Exception as e:
                print(f"⚠️ Error displaying {img_path}: {e}")

        lbl_page.config(text=f"Page {page + 1} / {total_pages.get()}")

    # --- Navigation control functions ---
    def next_page():
        if current_page.get() < total_pages.get() - 1:
            current_page.set(current_page.get() + 1)
            draw_grid()

    def prev_page():
        if current_page.get() > 0:
            current_page.set(current_page.get() - 1)
            draw_grid()

    def refresh_page(*_):
        current_page.set(0)
        draw_grid()

    # --- Bottom navigation controls ---
    nav_frame = tk.Frame(root, bg="#1e1e1e")
    nav_frame.pack(pady=10)

    ttk.Button(nav_frame, text="⬅️ Previous", command=prev_page).pack(side="left", padx=5)
    lbl_page = tk.Label(nav_frame, text="", fg="white", bg="#1e1e1e", font=("Segoe UI", 10, "bold"))
    lbl_page.pack(side="left", padx=5)
    ttk.Button(nav_frame, text="Next ➡️", command=next_page).pack(side="left", padx=5)
    ttk.Button(nav_frame, text="🔄 Refresh", command=refresh_page).pack(side="left", padx=10)
    ttk.Button(nav_frame, text="❌ Close", command=root.destroy).pack(side="left", padx=5)

    # --- Bind changes to grid size dropdown ---
    grid_menu.bind("<<ComboboxSelected>>", refresh_page)

    # --- Initial display ---
    draw_grid()

    root.mainloop()

# In[]
"""
def get_hf_model_img_labels_local_path(repoid, img_size, force_refresh=False):
    '''
    This function uses the above two functions multiclass_prediction_return, display_images_gui
    
    Description:
    ------------
    This script loads a pretrained model from Hugging Face, reads class names,
    runs predictions on test images, and displays results.
    
    Parameters
    ----------
    repoid : Hugging Face REPO_ID  where the model file, .txt class file, and a test image folder is stored 
    img_size : a tuple with two elements (300, 300). This value should match with the models input
    force_refresh = False will not redownload the model, class, and image files if they already exist locall
                  = True  will force a redownload of all model, class, and image files even if they already exist locally.

    Returns
    -------
    None.

    '''
    
    #IMG_SIZE = (160, 160)  #faster but lower accuracy don't change this size for MobileNetV2. model requires only this resolution
    IMG_SIZE = img_size  #inception_v3
    all_image_paths = []
    all_predicted_labels = []
    # -----------------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------------
    REPO_ID = repoid
    LOCAL_CACHE_DIR = "hf_insect_cache"
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    
    print(f"Listing files from: {REPO_ID}")
    files = list_repo_files(REPO_ID)
    for i, f in enumerate(files, start=1):
        print(f"{i:02d}. {f}")
    
    model_file = next((f for f in files if f.endswith(".keras")), None)  #identified the .keras file
    class_file = next((f for f in files if f.endswith(".txt")), None)  #identifies the .txt file in the repository
    
    
    # Identify test images — just filter by image extensions
    test_images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    
    model_full_path = hf_hub_download(repo_id=REPO_ID, filename=model_file, local_dir=LOCAL_CACHE_DIR, force_download=force_refresh)
    #The LOCAL_CACHE_DIR and the application both are in the same folder. The model is insdie the LOCAL_CACHE_DIR
    #So this is only relative path i.e only the model dir name
    class_file_full_path = hf_hub_download(repo_id=REPO_ID, filename=class_file, local_dir=LOCAL_CACHE_DIR, force_download=force_refresh)
    #This function downloads the class_file identified as above (endswith  .txt)  and returns the full path of that file
    with open(class_file_full_path, "r") as f:
        raw = f.read().strip()
    
    if "," in raw:
        class_names = [c.strip() for c in raw.split(",") if c.strip()]
    else:
        class_names = [line.strip() for line in raw.splitlines() if line.strip()]
    
    test_folder_path = os.path.join(LOCAL_CACHE_DIR, "InsectTest")
    os.makedirs(test_folder_path, exist_ok=True)
    
    downloaded_count = 0
    for img_file in test_images:
        local_img_path = hf_hub_download(repo_id=REPO_ID, filename=img_file, local_dir=LOCAL_CACHE_DIR, force_download=force_refresh)
        dest_path = os.path.join(test_folder_path, os.path.basename(local_img_path))
        if not os.path.exists(dest_path):
            shutil.copy(local_img_path, dest_path)
            downloaded_count += 1

    return model_full_path, test_folder_path, class_names
"""

# In[]

def get_hf_model_img_labels_local_path(REPO_ID, img_size, cache_folder="hf2kkt_download", force_refresh=False, model_subdir=None):
    '''
    This function uses the above two functions multiclass_prediction_return, display_images_gui
    
    Description:
    ------------
    This script loads a pretrained model from Hugging Face, reads class names,
    runs predictions on test images, and displays results.
    
    Parameters
    ----------
    REPO_ID : Hugging Face REPO_ID  where the model file, .txt class file, and a test image folder is stored 
    img_size : a tuple with two elements (300, 300). This value should match with the models input
    cache_folder: "hf2kkt_download"  default araguemnt may be omitted while calling
    force_refresh = False will not redownload the model, class, and image files if they already exist locall
                  = True  will force a redownload of all model, class, and image files even if they already exist locally.

    Returns
    -------
    downloaded model's loacal relative path  - hf2kkt_download\Insect_Inception_V3
    downloaded images folder's local relative path - hf2kkt_download\Insect_Inception_V3\InsectTest
    '''
    
    #IMG_SIZE = (160, 160)  #faster but lower accuracy don't change this size for MobileNetV2. model requires only this resolution
    IMG_SIZE = img_size  #inception_v3
    # -----------------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------------

    LOCAL_CACHE_DIR = cache_folder
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    
    print(f"Listing files from: {REPO_ID}")
    if model_subdir:
        # Filter only files inside the specific model folder
        files = [f for f in list_repo_files(REPO_ID) if f.startswith(f"{model_subdir}/")]
    else:
        files = list_repo_files(REPO_ID)

    for i, f in enumerate(files, start=1):
        print(f"{i:02d}. {f}")
    
    model_file = next((f for f in files if f.endswith(".keras")), None)  #identified the .keras file
    class_file = next((f for f in files if f.endswith(".txt")), None)  #identifies the .txt file in the repository
    
    
    # Identify test images — just filter by image extensions
    test_images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    
    downloaded_model_local_relative_path = hf_hub_download(repo_id=REPO_ID, filename=model_file, local_dir=LOCAL_CACHE_DIR, force_download=force_refresh)
    #The LOCAL_CACHE_DIR and the application both are in the same folder. The model is insdie the LOCAL_CACHE_DIR
    #So this is only relative path i.e only the model dir name
    class_file_full_path = hf_hub_download(repo_id=REPO_ID, filename=class_file, local_dir=LOCAL_CACHE_DIR, force_download=force_refresh)
    #This function downloads the class_file identified as above (endswith  .txt)  and returns the full path of that file
    with open(class_file_full_path, "r") as f:
        raw = f.read().strip()
    
    if "," in raw:
        class_names = [c.strip() for c in raw.split(",") if c.strip()]
    else:
        class_names = [line.strip() for line in raw.splitlines() if line.strip()]

    for img_file in test_images:
        local_img_path = hf_hub_download(repo_id=REPO_ID, filename=img_file, local_dir=LOCAL_CACHE_DIR, force_download=force_refresh)
    test_folder_path=os.path.dirname(local_img_path)

    return downloaded_model_local_relative_path, test_folder_path, class_names



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
