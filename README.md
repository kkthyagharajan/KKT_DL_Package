# 🧠 KKT_DL_Package

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

KKT_DL_Package is a modular Deep Learning utility library designed to simplify and accelerate the development of Keras and TensorFlow–based models. It integrates reusable components for transfer learning, dataset handling, callback management, and result visualization — making it ideal for rapid prototyping and research projects.

**KKT_DL_Package** is a modular Deep Learning utility library designed to simplify and accelerate model development using **Keras** and **TensorFlow**.  

## 🚀 Features

- 🧩 **Transfer Learning API**
- ⚙️ **Custom Training Callbacks**
- 📊 **Visualization Utilities**
- 🧮 **Dataset Utilities**
- 📦 **Modular Architecture**

## 🧩 Dependencies

This package requires the following core libraries and the following versions are recommended for best compatibility:

- **Python** ≥ 3.11  
- **TensorFlow** == 2.18.0  
- **OpenCV-Python** == 4.11.0.86  
- **OpenCV-Contrib-Python** == 4.11.0.86  
- **Pillow** == 11.1.0  
- **PyQt5** == 5.15.10  
- **CVLib** == 0.2.7  
- **Albumentations** == 1.3.1  
- **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**

## 🧑‍💻 Quick Example

```python
import KKT_DL_Package as kkt

model = kkt.KKT_Keras_API_TransferLearning_Models.build_model(
    base_model='MobileNetV2',
    num_classes=3,
    input_shape=(224, 224, 3)
)

callbacks = kkt.KKT_Callback_Functions.get_callbacks(save_path='models/')
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
kkt.KKT_plots.plot_training_curves(history)
```

## 🧾 Installation
### 1️⃣ For **PyPI** users (public install)
The following command will install the package along with all required dependencies in your environment.

```bash
pip install KKT_DL_Package
```
### 2️⃣ For developers / local editable install
If you are a developer install from GitHub source using the following commands. 
This also automatically installs all dependencies.
```bash
git clone https://github.com/kkthyagharajan/KKT_DL_Package.git
cd KKT_DL_Package
pip install -e .
```
This will clone the `KKT_DL_Package` repository into your **current working directory**, including all subfolders and files.

### 3️⃣ Manual dependency install (Optional)
If you want to install only dependencies without installing the code

```bash
pip install -r requirements.txt
```


## 📜 License

This project is licensed under the [MIT License](LICENSE).

## ⚙️ Acknowledgments

This package builds upon several open-source libraries, including:

- **TensorFlow** — Apache License 2.0  
- **Keras** — Apache License 2.0  
- **NumPy** — BSD License  
- **Pandas** — BSD License  
- **Matplotlib** — PSF License  
- **Pillow** — HP License  

I gratefully acknowledge the developers and contributors of these projects.

## 👤 Author
**Thyagharajan K K**



