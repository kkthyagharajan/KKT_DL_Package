# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 14:02:16 2025

@author: THYAGHARAJAN

V2 Provides GUI, but takes minimum image count for balancing
V3 Augmentation was included
V4 Augmentation methods are displayed below the monitor window. But not working
V5 displays the parent folder summary
V6 augmented images are stored in the target dir with its originalname_aug_some random number
"""

# Tkinter GUI for creating a balanced dataset for training, validation, and testing.

import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import defaultdict
from PIL import Image, ImageEnhance, ImageOps
import random
import math

def augment_and_save_image(img_path, output_dir, augmentation_methods, img_name_prefix="aug"):
    """Applies selected augmentations to an image and saves the result with a prefix."""
    try:
        img = Image.open(img_path)
        img_name, img_ext = os.path.splitext(os.path.basename(img_path))
        
        # Apply selected augmentations
        if "Random Flip" in augmentation_methods:
            if random.choice([True, False]):
                img = ImageOps.mirror(img)
            if random.choice([True, False]):
                img = ImageOps.flip(img)
        
        if "Random Rotation" in augmentation_methods:
            angle = random.uniform(-20, 20)
            img = img.rotate(angle, resample=Image.Resampling.BICUBIC)

        if "Random Brightness" in augmentation_methods:
            brightness_factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)

        if "Random Contrast" in augmentation_methods:
            contrast_factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
            
        if "Random Zoom" in augmentation_methods:
            width, height = img.size
            zoom_factor = random.uniform(0.8, 1.2)
            new_width = int(width / zoom_factor)
            new_height = int(height / zoom_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Crop or pad to original size
            left = (new_width - width) / 2
            top = (new_height - height) / 2
            right = (new_width + width) / 2
            bottom = (new_height + height) / 2
            img = img.crop((left, top, right, bottom))
            img = img.resize((width, height), Image.Resampling.LANCZOS)

        # Save the augmented image with the specified prefix
        new_filename = f"{img_name_prefix}_{img_name}_aug_{random.randint(1000,9999)}{img_ext}"
        img.save(os.path.join(output_dir, new_filename))

    except Exception as e:
        print(f"Error augmenting image {img_path}: {e}")

def create_balanced_dataset(source_dir, test_ratio, validation_ratio, train_ratio,
                            output_text_widget, status_bar_label, should_shuffle, balance_method, augmentation_methods):
    """   
    Splits dataset into train/validation/test and returns their paths + class names.

    Args:
        input_dir (str): Path to the raw dataset folder (with subfolders = classes).
        output_dir (str): Path where split folders will be created.
        split (tuple): Fractions for train, val, test.
    
    """
    try:
        status_bar_label.config(text="Status: Starting dataset creation...")
        output_text_widget.insert(tk.END, "Please wait, the process is running...\n")
        output_text_widget.update_idletasks()

        if not (0.99 <= (test_ratio + validation_ratio + train_ratio) <= 1.01):
            messagebox.showerror("Error", "The sum of ratios must be approximately 1.0.")
            status_bar_label.config(text="Status: Error")
            return

        class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        if not class_names:
            messagebox.showerror("Error", "No class folders found in the source directory.")
            status_bar_label.config(text="Status: Error")
            return
        
        all_class_images = defaultdict(list)
        total_images_count = 0
        for class_name in class_names:
            class_path = os.path.join(source_dir, class_name)
            images = [
                os.path.join(class_path, f) for f in os.listdir(class_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            all_class_images[class_name].extend(images)
            total_images_count += len(images)
        
        if total_images_count == 0:
            messagebox.showerror("Error", "No images found in any class folders.")
            status_bar_label.config(text="Status: Error")
            return
            
        output_text_widget.insert(tk.END, "\n--- Starting Splitting Process ---\n\n")
        output_text_widget.update_idletasks()

        source_name = os.path.basename(source_dir)
        target_dir = os.path.join(os.path.dirname(source_dir), f"split_{source_name}")
        
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        os.makedirs(target_dir, exist_ok=True)
        
        split_dirs = {
            "train": os.path.join(target_dir, "train"),
            "valid": os.path.join(target_dir, "valid"),
            "test": os.path.join(target_dir, "test")
        }
        
        for split_path in split_dirs.values():
            os.makedirs(split_path, exist_ok=True)

        for class_name in class_names:
            for split_path in split_dirs.values():
                class_path = os.path.join(split_path, class_name)
                os.makedirs(class_path, exist_ok=True)
        
        status_bar_label.config(text="Status: Collecting images and preparing split...")
        output_text_widget.update_idletasks()
        
        if balance_method == "Minimum Count Image Balance":
            target_count = min(len(images) for images in all_class_images.values())
            output_text_widget.insert(tk.END, f"Balancing method: Minimum Count. Target per class: {target_count}\n\n")
        else:
            target_count = max(len(images) for images in all_class_images.values())
            output_text_widget.insert(tk.END, f"Balancing method: Augmented Image. Target per class: {target_count}\n\n")
            
        output_text_widget.update_idletasks()

        summary_counts = {
            "train": defaultdict(int),
            "valid": defaultdict(int),
            "test": defaultdict(int)
        }
        
        for class_name, image_paths in all_class_images.items():
            status_bar_label.config(text=f"Status: Processing class '{class_name}'...")
            output_text_widget.update_idletasks()

            current_count = len(image_paths)
            
            # Augment images if necessary and add them to the list of images to be split
            if balance_method == "Augmented Image Balance" and current_count < target_count:
                num_to_augment = target_count - current_count
                output_text_widget.insert(tk.END, f"  Augmenting {num_to_augment} images for class '{class_name}'...\n")
                output_text_widget.update_idletasks()
                
                if not image_paths:
                    output_text_widget.insert(tk.END, f"  Warning: No images to augment for class '{class_name}'. Skipping.\n")
                    continue
                    
                # Store augmented images in a list for later splitting
                for _ in range(num_to_augment):
                    img_to_augment = random.choice(image_paths)
                    dest_path = os.path.join(split_dirs["train"], class_name)
                    augment_and_save_image(img_to_augment, dest_path, augmentation_methods)

            final_images = []
            if balance_method == "Augmented Image Balance":
                final_images.extend(image_paths)
                augmented_paths = [os.path.join(split_dirs["train"], class_name, f) 
                                   for f in os.listdir(os.path.join(split_dirs["train"], class_name))
                                   if f.startswith("aug_")]
                final_images.extend(augmented_paths)
            else:
                final_images = image_paths[:target_count]

            if should_shuffle:
                random.shuffle(final_images)

            total_images = len(final_images)
            train_count = math.floor(total_images * train_ratio)
            validation_count = math.floor(total_images * validation_ratio)
            test_count = total_images - train_count - validation_count

            train_set = final_images[:train_count]
            validation_set = final_images[train_count:train_count + validation_count]
            test_set = final_images[train_count + validation_count:]

            for img_list, split_name in [(train_set, "train"), (validation_set, "valid"), (test_set, "test")]:
                for img_path in img_list:
                    # Don't copy if it's an augmented image already in the train folder
                    if split_name == "train" and "aug_" in os.path.basename(img_path):
                        shutil.move(img_path, os.path.join(split_dirs[split_name], class_name, os.path.basename(img_path)))
                        summary_counts[split_name][class_name] += 1
                    else:
                        dest_path = os.path.join(split_dirs[split_name], class_name, os.path.basename(img_path))
                        shutil.copy(img_path, dest_path)
                        summary_counts[split_name][class_name] += 1
        
        output_text_widget.insert(tk.END, "\n--- Final Split Summary ---\n\n")
        for split_name, counts in summary_counts.items():
            output_text_widget.insert(tk.END, f"--- {split_name.capitalize()} Dataset ---\n")
            for class_name, count in sorted(counts.items()):
                output_text_widget.insert(tk.END, f"  Class: {class_name}, Images: {count}\n")
            output_text_widget.insert(tk.END, "\n")
            
        status_bar_label.config(text="Status: Done")
        messagebox.showinfo("Success", "Dataset splitting complete!")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        status_bar_label.config(text="Status: Error")

class DatasetSplitterApp:
    def __init__(self, master):
        self.master = master
        master.title("Balanced Dataset Splitter")
        master.geometry("600x650")
        
        self.source_dir = tk.StringVar()
        self.shuffle_var = tk.BooleanVar(value=True)
        self.balance_method_var = tk.StringVar(value="Minimum Count Image Balance")
        self.aug_methods = {
            "Random Flip": tk.BooleanVar(),
            "Random Rotation": tk.BooleanVar(),
            "Random Brightness": tk.BooleanVar(),
            "Random Contrast": tk.BooleanVar(),
            "Random Zoom": tk.BooleanVar()
        }

        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.master, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        dir_frame = tk.Frame(main_frame)
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(dir_frame, text="Source Data Directory:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.source_entry = tk.Entry(dir_frame, textvariable=self.source_dir, width=40)
        self.source_entry.pack(side=tk.LEFT, padx=(5, 0), expand=True, fill=tk.X)
        tk.Button(dir_frame, text="Browse", command=self.select_source_dir).pack(side=tk.LEFT, padx=(5, 0))

        ratio_frame = tk.Frame(main_frame)
        ratio_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(ratio_frame, text="Split Ratios (e.g., 0.7, 0.15, 0.15):", font=("Helvetica", 10, "bold")).pack(anchor="w")

        test_frame = tk.Frame(ratio_frame)
        test_frame.pack(fill=tk.X, pady=(5, 0))
        tk.Label(test_frame, text="Test:", width=10, anchor="w").pack(side=tk.LEFT)
        self.test_ratio_entry = tk.Text(test_frame, height=1, width=5)
        self.test_ratio_entry.insert("1.0", "0.2")
        self.test_ratio_entry.pack(side=tk.LEFT)

        val_frame = tk.Frame(ratio_frame)
        val_frame.pack(fill=tk.X, pady=(5, 0))
        tk.Label(val_frame, text="Validation:", width=10, anchor="w").pack(side=tk.LEFT)
        self.val_ratio_entry = tk.Text(val_frame, height=1, width=5)
        self.val_ratio_entry.insert("1.0", "0.2")
        self.val_ratio_entry.pack(side=tk.LEFT)
        
        train_frame = tk.Frame(ratio_frame)
        train_frame.pack(fill=tk.X, pady=(5, 0))
        tk.Label(train_frame, text="Train:", width=10, anchor="w").pack(side=tk.LEFT)
        self.train_ratio_entry = tk.Text(train_frame, height=1, width=5)
        self.train_ratio_entry.insert("1.0", "0.6")
        self.train_ratio_entry.pack(side=tk.LEFT)
        
        tk.Checkbutton(main_frame, text="Shuffle data before splitting", variable=self.shuffle_var).pack(anchor="w", pady=(10, 5))

        balance_frame = tk.Frame(main_frame)
        balance_frame.pack(fill=tk.X, pady=(10, 5))
        tk.Label(balance_frame, text="Balancing Method:", font=("Helvetica", 10, "bold")).pack(anchor="w")
        
        radio_frame = tk.Frame(balance_frame)
        radio_frame.pack(fill=tk.X)
        tk.Radiobutton(radio_frame, text="Minimum Count Image Balance", variable=self.balance_method_var,
                       value="Minimum Count Image Balance", command=self.toggle_aug_checkboxes).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Radiobutton(radio_frame, text="Augmented Image Balance", variable=self.balance_method_var,
                       value="Augmented Image Balance", command=self.toggle_aug_checkboxes).pack(side=tk.LEFT)
        
        self.aug_checkbox_frame = tk.Frame(main_frame)
        tk.Label(self.aug_checkbox_frame, text="Augmentation Methods:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.aug_checkboxes = []
        for text, var in self.aug_methods.items():
            cb = tk.Checkbutton(self.aug_checkbox_frame, text=text, variable=var)
            cb.pack(side=tk.LEFT, padx=5)
            self.aug_checkboxes.append(cb)
        
        self.toggle_aug_checkboxes()

        action_frame = tk.Frame(main_frame, pady=10)
        action_frame.pack()
        tk.Button(action_frame, text="Create Balanced Dataset", command=self.start_splitting, font=("Helvetica", 12, "bold")).pack()
        
        monitor_label = tk.Label(main_frame, text="Monitor", font=("Helvetica", 10, "bold"), anchor="w")
        monitor_label.pack(fill=tk.X, pady=(10, 0))

        self.summary_text = tk.Text(main_frame, height=15, width=60, padx=5, pady=5)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        self.status_bar_label = tk.Label(self.master, text="Status: Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar_label.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_aug_checkboxes(self):
        """Shows or hides augmentation checkboxes based on radio button selection."""
        if self.balance_method_var.get() == "Augmented Image Balance":
            self.aug_checkbox_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.aug_checkbox_frame.pack_forget()

    def select_source_dir(self):
        """Opens a file dialog to select the source directory and displays a summary."""
        directory = filedialog.askdirectory()
        if directory:
            self.source_dir.set(directory)
            self.show_initial_summary()

    def show_initial_summary(self):
        """Displays a summary of the selected directory's contents."""
        source_dir = self.source_dir.get()
        if not source_dir or not os.path.isdir(source_dir):
            return

        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, "Initial Dataset Summary:\n")
        self.summary_text.insert(tk.END, f"Source Directory: {source_dir}\n\n")

        total_images = 0
        try:
            class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
            if not class_names:
                self.summary_text.insert(tk.END, "No class folders found.\n")
                return

            for class_name in class_names:
                class_path = os.path.join(source_dir, class_name)
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                num_images = len(images)
                self.summary_text.insert(tk.END, f"  Class '{class_name}': {num_images} images\n")
                total_images += num_images

            self.summary_text.insert(tk.END, f"\nTotal images found: {total_images}\n")
        except Exception as e:
            self.summary_text.insert(tk.END, f"An error occurred while scanning the directory: {e}\n")


    def start_splitting(self):
        """Validates inputs and starts the dataset splitting process."""
        source_dir = self.source_dir.get()
        if not source_dir or not os.path.isdir(source_dir):
            messagebox.showerror("Error", "Please select a valid source directory.")
            return

        try:
            test_ratio = float(self.test_ratio_entry.get("1.0", tk.END).strip())
            val_ratio = float(self.val_ratio_entry.get("1.0", tk.END).strip())
            train_ratio = float(self.train_ratio_entry.get("1.0", tk.END).strip())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for the ratios.")
            return

        should_shuffle = self.shuffle_var.get()
        balance_method = self.balance_method_var.get()
        augmentation_methods = [
            method for method, var in self.aug_methods.items() if var.get()
        ]

        if balance_method == "Augmented Image Balance" and not augmentation_methods:
            messagebox.showerror("Error", "Please select at least one augmentation method for the 'Augmented Image Balance' option.")
            return
            
        create_balanced_dataset(source_dir, test_ratio, val_ratio, train_ratio,
                                self.summary_text, self.status_bar_label, should_shuffle, balance_method, augmentation_methods)

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetSplitterApp(root)
    root.mainloop()