<!-- ──────────────────────────────────────────────── -->
<!-- 🎖️ Banner Section -->
<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.x-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
</p>

<h1 align="center">🧮 KKT_Keras_Balanced_Dataset</h1>
<p align="center">
  <strong>A GUI-integrated dataset balancing utility module within <code>KKT_DL_Package</code></strong><br>
  Automatically balances image datasets for deep learning using oversampling or undersampling strategies.
</p>
<hr style="border: 1px solid #ccc;">
<!-- ──────────────────────────────────────────────── -->

## ⚙️ Features
- 📊 **Automatic Class Balancing** — Equalizes image counts across all class folders.  
- 🔀 **Two Balancing Strategies:**
  - **Oversampling (Augment to Maximum Class Size)** — Duplicates or augments smaller classes.  
  - **Undersampling (Reduce to Minimum Class Size)** — Randomly trims larger classes.  
- 🎨 **Built-in Augmentation Support** — Uses internal augmentation utilities for more diverse data.  
- 🗂️ **Automatic Output Folder Creation** — Balanced dataset automatically generated next to the input folder.  
- 🧩 **Train / Validation / Test Splitting** — Output is pre-structured for training pipelines.  
- 🖥️ **Standalone GUI Tool** — Easy interface for non-programmatic use.

---

## 📂 Source Directory Structure
Your dataset should follow a **class-based folder layout**, such as:

<source_dir_name>/
│
├── class_A/
│ ├── img_001.jpg
│ ├── img_002.jpg
│ └── ...
├── class_B/
│ ├── img_101.jpg
│ └── ...
└── class_C/
├── img_201.jpg
└── ...

Each subfolder corresponds to a distinct class.

---

## 📦 Output Directory Structure (Automatically Created)

When executed, the module automatically creates a new folder named  
**`split_<source_dir_name>`** in the **same directory as the input**.  
Inside this folder, the data is divided into **train**, **valid**, and **test** subsets —  
each containing balanced class folders.

Example:
split_<source_dir_name>/
│
├── train/
│ ├── class_A/
│ ├── class_B/
│ └── class_C/
│
├── valid/
│ ├── class_A/
│ ├── class_B/
│ └── class_C/
│
└── test/
├── class_A/
├── class_B/
└── class_C/

If the input folder is: D:\Datasets\COVID_Xray
then the program automatically creates: D:\Datasets\split_COVID_Xray\


✅ Each subset (train / valid / test) contains balanced samples per class based on your chosen strategy:
- Oversampling (Augment to Maximum Class Size)
- Undersampling (Reduce to Minimum Class Size)

---

## 🚀 Usage Options

### 🧠 1. Programmatic Usage
```python
from KKT_DL_Package.utils import KKT_Keras_Balanced_Dataset

# Balance dataset using oversampling (with augmentation)
KKT_Keras_Balanced_Dataset.balance_dataset(
    input_dir="dataset/",
    method="oversample",  # or "undersample"
    augment=True
)


🖥️ 2. Standalone GUI Mode

You can also launch this module directly from the command line:
python KKT_Keras_Balanced_Dataset.py

GUI Features:
Select Input Directory (source dataset)
Choose Balancing Strategy:
 Oversampling (Augment to Maximum Class Size)
 Undersampling (Reduce to Minimum Class Size)

Click Start Balancing
Progress updates and a completion message are displayed in the interface.

🧠 Notes
Supported formats: .jpg, .jpeg, .png
Output folder split_<source_dir_name> is auto-generated next to the source folder.
Existing directories with the same name will be reused or replaced (based on implementation).
Works on Windows, Linux, and macOS.
Fully compatible with Keras, TensorFlow, and other DL frameworks.

👤 Author
Thyagharajan K K
📧 kkthyagharajan@yahoo.com