<<<<<<< HEAD
# 🧠 KKT_DL_Package
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**KKT_DL_Package** is a modular Deep Learning utility library designed to simplify and accelerate model development using **Keras** and **TensorFlow**.  
It includes flexible tools for **transfer learning**, **custom callbacks**, **balanced dataset creation**, and **visualization**.

## 🚀 Features

- 🧩 **Transfer Learning API**
- ⚙️ **Custom Training Callbacks**
- 📊 **Visualization Utilities**
- 🧮 **Dataset Utilities**
- 📦 **Modular Architecture**

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

```bash
pip install KKT_DL_Package
```

or from source:

```bash
git clone https://github.com/kkthyagharajan/KKT_DL_Package.git
cd KKT_DL_Package
pip install -e .
```

## 📋 Requirements

- Python ≥ 3.8
- TensorFlow ≥ 2.9
- NumPy, Pandas, Matplotlib, Pillow

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
=======
# KKT_DL_Package
KKT_DL_Package is a modular Deep Learning utility library designed to simplify and accelerate the development of Keras and TensorFlow–based models. It integrates reusable components for transfer learning, dataset handling, callback management, and result visualization — making it ideal for rapid prototyping and research projects.
>>>>>>> 2d5c7f28dd57af6a449e594b1377a38f079dd8e2
