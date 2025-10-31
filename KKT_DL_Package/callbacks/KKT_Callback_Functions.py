# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 10:03:11 2025

@author: THYAGHARAJAN
"""
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os


# In[61]
# Define the EarlyStopping callback
def KKT_early_stopping(patience):
    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',     # Metric to monitor (you can change it to 'val_accuracy' or other metrics)
        patience=patience,             # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True   # Restores the model weights from the epoch with the best value of the monitored metric
    )

# In[61]:
   
def Reduce_Lr(lr_reduction_factor, learning_rate_patience):
    # Learning Rate Scheduling: Implement learning rate scheduling to gradually reduce the learning rate during training. 
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=lr_reduction_factor,
        patience=learning_rate_patience,
        min_lr=1e-9)
    #This does not stop training only adjusts the learning rate. It gives the optimizer a chance to get a better minimum

# In[]:
'''
    # Simple attention function
def attention_block(inputs):
    # Compute the attention scores
    W = tf.keras.layers.Dense(inputs.shape[-1], activation='tanh')(inputs)
    score = tf.keras.layers.Dense(1)(W)
    attention_weights = tf.nn.softmax(score, axis=1)

    # Apply the attention weights to the inputs
    context_vector = attention_weights * inputs
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector


'''

# In[62]
def KKT_Save_Model():
    model_save_dir=r"D:\Python_Spyder_Working Dir\1 All program summary\KKT_Models\MobileNetV2"
    # Create the directory if it doesn't exist to save the model
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    #It does not stop training. At the end of each epoch, it checks if the current val_accuracy is the best seen so far. 
    #If it is, it saves the model.
    modelSave_checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_save_dir, "model_{epoch:02d}_{val_accuracy:.4f}.keras"),
        monitor="val_accuracy",  # Save based on validation accuracy
        save_best_only=True,      # Only save the best model based on the monitored metric
        save_weights_only=False,  # Save the entire model (including architecture) rather than just weights
        verbose=1
    )



