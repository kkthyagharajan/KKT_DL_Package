# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 07:10:49 2025

@author: THYAGHARAJAN
"""
'''
Last change was made on 22-1-2025 no_of_legend_col parameter in plot_learning_curves
'''
# Not working to be checked the batch size used may be different
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

# In[]:
'''
Both the Confusion_matrix functions produce same results. In the second case title should not be passed as keyword argument
KKT_plots.Confusion_matrix_class_number(KKT_model, test_dataset, title="Confusion Matrix: Before Fine-Tuning")
KKT_plots.Confusion_matrix_class_names(KKT_model, test_dataset,"Confusion Matrix: Before Fine-Tuning",class_names)
KKT_plots.Multiclass_confusion_matrix
#you can plot confusion matrix for test_dataset or validation dataset
'''

# In[]:
# Use KKT_model & use test_dataset
# Assuming KKT_model is already trained
#predicts as class 0 , class 1
def Confusion_matrix_class_number(KKT_model, test_dataset,title):
    predictions = KKT_model.predict(test_dataset)
    # probability value for each sample in the test_dataset
    predicted_labels = (predictions > 0.5).astype(int)  #predicts as class 0 , class 1
    true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# In[]:
#you can plot confustion matrix for test_dataset or validation_dataset
#uses class names
def Confusion_matrix_class_names(KKT_model, dataset, class_names, title):  #title is keyword argumnent should come after positional arguments
    # Predict on the dataset. title should not be passes as keyword argument
    y_true = []
    y_pred = []
    
    for images, labels in dataset:
        predictions = tf.sigmoid(KKT_model(images))  # Apply sigmoid activation
        predictions = tf.round(predictions).numpy().astype(int).flatten()
        y_pred.extend(predictions)
        y_true.extend(labels.numpy().astype(int).flatten())
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# In[]:

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def Multiclass_confusion_matrix_with_report(KKT_model, target_subdirs, dataset):
    """
    Evaluates the model and combines confusion matrix with precision, recall, and F1-score.
    
    Parameters:
    - KKT_model: Trained Keras KKT_model for predictions.
    - target_subdirs: List of class names used by the KKT_model.
    - dataset: Evaluation dataset (batched and preprocessed).
        
    Returns:
    - Confusion Matrix and Class-Specific Metrics
    """
    # Extract true labels and predictions
    y_true = np.concatenate([y for _, y in dataset], axis=0)  # Concatenate true labels
    y_pred_probs = KKT_model.predict(dataset)  # Predict probabilities
    y_pred = np.argmax(y_pred_probs, axis=1)  # Get predicted class indices

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Plot the confusion matrix   
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_subdirs)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
    # Classification Report
    print("\nClassification Report:\n")
    report = classification_report(y_true, y_pred, target_names=target_subdirs)
    print(report)
    

# In[]:
def plot_learning_curves(training_acc, validation_acc, training_loss, validation_loss, no_of_legend_col, folds_endEpoch, title,zoom=1):    
  
    
    plt.figure(figsize=(8, 8))
    
    plt.subplot(2, 1, 1) #2rows 1 column, position 1
    plt.plot(range(1, len(training_acc)+1),training_acc, label='Training Accuracy')
    plt.plot(range(1,len(validation_acc)+1),validation_acc, label='Validation Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    if zoom==1:
        plt.ylim([min(plt.ylim()),1])
    else:
        plt.ylim([0,1])
        
    if len(folds_endEpoch) > 0:
        fold_no=1
        for i in folds_endEpoch: 
            plt.plot([i, i], plt.ylim(), label=f'End of Fold {fold_no}, epcoch {i}' , linestyle='--')  # Draw the line
            # Add the label near the line
            fold_no+=1
            plt.text(i, plt.ylim()[1] * 0.95, f'Epoch {i}', rotation=90, verticalalignment='top', fontsize=8, color='blue')
        
    plt.legend(loc='upper center', bbox_to_anchor=(0.4, -0.25), ncol=no_of_legend_col, frameon=False, handlelength=1.5, columnspacing=0.5)  # Adjusted y-value and reduced legend width
    plt.title('Training and Validation Accuracy -' + title)
    
    
    plt.subplot(2, 1, 2)  #plot for position 2 
    plt.plot(range(1, len(training_loss)+1),training_loss, label='Training Loss')
    plt.plot(range(1,len(validation_loss)+1),validation_loss, label='Validation Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    if zoom == 1:
        plt.ylim([min(training_loss + validation_loss), max(training_loss + validation_loss)])  
        #training_loss and validation_loss are lists. Dynamically adjust limits
    else:
        plt.ylim([0, max(training_loss + validation_loss)])     
        
    if len(folds_endEpoch) > 0:
        fold_no=1
        for i in folds_endEpoch:
            plt.plot([i, i], plt.ylim(), label=f'End of Fold {fold_no} , epcoch {i}', linestyle='--')  # Draw the line
            # Add the label near the line
            fold_no +=1
            plt.text(i, plt.ylim()[1] * 0.95, f'Epoch {i}', rotation=90, verticalalignment='top', fontsize=8, color='blue')
                  
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.4, -0.25), ncol=no_of_legend_col, frameon=False, handlelength=1.5, columnspacing=0.5)  # Adjusted y-value and reduced legend width
    plt.title('Training and Validation Loss - ' + title)
    
    plt.subplots_adjust(hspace=0.9) 
    plt.show()

# In[]:
#GANTT CHART  Gantt Chart
    

# Define tasks and their durations (start month, end month)
tasks = [
    ("Planning & Research", 1, 3),  # No change
    ("Basic Hardware Setup & Testing", 4, 6),  # No change
    ("Integrating IR & PIR Sensors", 7, 9),  # No change
    ("Testing IR & PIR Sensors", 10, 12),  # No change
    ("LoRaWAN Integration", 13, 15),  # No change
    ("Communication Setup", 16, 18),  # No change
    ("Adding Renewable Energy Sources", 19, 21),  # No change
    ("Repellent Mechanism Implementation", 22, 24),  # No change
    ("Field Deployment & Initial Testing", 25, 27),  # No change
    ("System Refinement & Performance Enhancement", 28, 30),  # No change
    ("Large-Scale Deployment & User Training", 31, 33),  # No change
    ("Impact Assessment & Documentation", 34, 36)  # No change
]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for different phases
colors = plt.cm.viridis(np.linspace(0, 1, len(tasks)))


    
# Plot Gantt bars (no changes in logic, but important to use 'align='edge')
for i, (task, start, end) in enumerate(tasks):
    ax.barh(i, end - start + 1, left=start -1 , height=0.8, color=colors[i], edgecolor="black", align='edge')  # Subtract 1 from 'left'


# Set y-axis labels to task names (in original order)
ax.set_yticks(np.arange(len(tasks)))
ax.set_yticklabels([task[0] for task in tasks])

# Set x-axis ticks and labels (CUSTOM VALUES)
x_ticks = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]  # Your custom values
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks)

# Draw vertical milestone lines (CUSTOM VALUES and red color)
for month in x_ticks:  # Use the same custom values for vertical lines
    ax.axvline(month, color="red", linestyle="--", alpha=0.7)



# Labels and title
ax.set_xlabel("Months")
ax.set_ylabel("Project Phases")
plt.figtext(0.5, 0.01, "Gantt Chart for Project Timeline", ha="center", fontsize=12, fontweight="bold")  # Added at the bottom


# Grid
ax.grid(axis="x", linestyle="--", alpha=0.5)

# Show the plot
plt.show()   
    
# In[]:
    
    
'''
#delete this histories may not be combined
def plot_learning_curves2(fit_history, finetune_start_epoch=0, zoom=1, title1='Base Model Frozen'):
    #plot_accuracy_loss_VS_epochs
    #for head_train_history, the default value 0 may be used, it indicates that no green line is to be drawn. 
    #for fine tuning, the initial epoch will be number of previous seesion epochs+1 i.e epoch of start of the second session
    
    plt.figure(figsize=(8, 8))
    ax1=plt.subplot(2, 1, 1)
    
    ax1.plot(range(1,len(fit_history.history['accuracy'])+1),fit_history.history['accuracy'], label='Training Accuracy')
    ax1.plot(range(1,len(fit_history.history['val_accuracy'])+1),fit_history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Accuracy')
    if zoom==1:
        ax1.set_ylim([min(ax1.get_ylim()),1])
    else:
        ax1.set_ylim([0,1])
        
    if finetune_start_epoch!=0:
        plt.plot([finetune_start_epoch-1,finetune_start_epoch-1],       #fine tunining's initial epoch - 1 = previous session's last epoch
                  plt.ylim(), label='Start Fine Tuning')
        
    ax1.legend(loc='lower right')

    
    ax1.text(0.5, -0.15, 'Training and Validation Accuracy - ' + title1,
         fontdict={'fontname': 'Times New Roman', 'fontsize': 10},
         ha='center', va='center', transform=ax1.transAxes)

    ax1.set_title('Training and Validation Accuracy - ' + title1,
                  fontdict={'fontname': 'Times New Roman', 'fontsize': 10}, pad=20)

    
    ax2=plt.subplot(2, 1, 2)
    ax2.plot(range(1,len(fit_history.history['loss'])+1), fit_history.history['loss'], label='Training Loss')
    ax2.plot(range(1,len(fit_history.history['val_loss'])+1),fit_history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Loss')  #cross entropy is used for binary classification
    if zoom==1:
        ax2.set_ylim([min(ax2.get_ylim()),1])
    else:
        ax2.set_ylim([0,1])
        
    if finetune_start_epoch!=0:
        plt.plot([finetune_start_epoch-1,finetune_start_epoch-1],       #fine tunining's initial epoch - 1 = previous session's last epoch
                  plt.ylim(), label='Start Fine Tuning')

    ax2.legend(loc='upper right')
    ax2.set_title('Training and Validation Loss - ' + title1,
                  fontdict={'fontname': 'Times New Roman', 'fontsize': 10}, pad=20)

    plt.subplots_adjust(hspace=0.4) 
    plt.show()
    
'''
# In[]:
'''
class_names = train_dataset.class_names #2 class names cats & dogs
plt.figure(figsize=(10, 10))   #figure size is 10x10 inches when saved on a file
for images, labels in train_dataset.take(1):      #take the first batch of images from the train_dataset
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)   #create 3x3 grid
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")     #hides the axis ticks
    
# In[11]:
#displays augmented images from training dataset
for image, _ in train_dataset.take(1):     #retrieves one batch of images from the train_dataset
# '_ variable'  is used to ignore the corresponding labels of the images in the batch
  plt.figure(figsize=(10, 10))  #image size 10x10 inches
  first_image = image[0]
  for i in range(9):   #9 augmented images
    ax = plt.subplot(3, 3, i + 1)  #3 rows 3 col (i+1)th row occupied
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))  #uses data augmentation pipeline
    #adds an extra dimension to the image tensor to match the expected input shape of the augmentation pipeline.
    plt.imshow(augmented_image[0] / 255)  #scales the pixel values to the range [0, 1] for proper visualization.
    plt.axis('off') #turns off the axis labels and ticks
    
# In[15]:
image_batch, label_batch = next(iter(train_dataset))
#This line retrieves a batch of images and their corresponding labels from the train_dataset. 
feature_batch = base_model(image_batch)
#passes the image_batch through the base_model, 
#transforming the input images into a higher-dimensional feature representation.

# In[27]:
# ### Learning curves
train_acc_history = history_1.history['accuracy']    #accuracy is a list, but loss is not stored in a list
val_acc_history = history_1.history['val_accuracy']

train_loss_history = history_1.history['loss']
val_loss_history = history_1.history['val_loss']  #validation loss


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(train_acc_history)) + 1, train_acc_history, label='Training Accuracy') 
#x-axis ticks will start at 0 to make it to start from 1, give the x values and then y values then label for the graph
plt.plot(np.arange(len(val_acc_history)) + 1, val_acc_history, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])  # here maximum value in y axis is set to 1
plt.title('Training and Validation Accuracy')
plt.xlabel('epoch')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(train_loss_history)) + 1, train_loss_history, label='Training Loss')
#x-axis ticks will start at 0 to make it to start from 1, give the x values and then y values then label for the graph
plt.plot(np.arange(len(val_loss_history)) + 1, val_loss_history, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([min(plt.ylim()), max(plt.ylim())])  # finds min & max values for both graphs and then plots
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing as needed
plt.show()

# In[35]:


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy - After Fine Tuning')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss - After Fine Tuning')
plt.xlabel('epoch')
plt.show()

# In[]:
# Check class names. Not needed
class_names = train_dataset.class_names
#train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
print(class_names)  # This will print the list of class names in the order they are encoded
#The labels provided by image_dataset_from_directory are integers, 
#making it compatible with SparseCategoricalCrossentropy used in model compilation

# In[3]:
# Functions
#epoch number should be given explicitly to plot function because default value is 0
# accuracy & loss plots, confusion matrix functions    
def plot_history(acc,val_acc,loss,val_loss,plot1,plot2):   
    epoch_no_arr= np.arange(1, len(acc)+1)   
    #here since the epoch number has to start from 1 in the plot but 'acc' array starts with index 0
    #plt.figure(figsize=(8, 8))
    plt.plot(plot1)
    #plt.subplot(2, 1, 1)
    plt.plot(epoch_no_arr, acc, label='Training Accuracy')
    plt.plot(epoch_no_arr, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    
    if len(acc)>initial_epochs:
        plt.plot([initial_epochs,initial_epochs],                 #Plot the vertical line at initial_epochs (10)
              plt.ylim(), label='Start Fine Tuning')
    
    plt.legend(loc='lower right')
    #plt.title('Training and Validation Accuracy')
    plt.xlim([1, len(acc)])  #loss curves start from 1
    plt.xlabel('epoch')    #use this if you take a single plot
    plt.show()
    
    plt.plot(plot2)
    #plt.subplot(2, 1, 2)
    plt.plot(epoch_no_arr, loss, label='Training Loss')
    plt.plot(epoch_no_arr,val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    if len(acc)>initial_epochs:
        plt.plot([initial_epochs,initial_epochs],                 #Plot the vertical line at initial_epochs (10)
              plt.ylim(), label='Start Fine Tuning')
    
    plt.legend(loc='upper right')
    #plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.xlim([1, len(val_loss)])  #loss curves start from 1 and ends based on number epochs i.e len(val_loss)
    plt.show()
    
# In[42]:
#Uses the functions given in [41] and plot the history graphs and confusion matrix

#plot accuracy and loss graphs
plot_history(acc,val_acc,loss,val_loss,1,2)


# Plot confusion matrix
validation_generator.reset()
y_true = validation_generator.classes
y_pred = model.predict(validation_generator)  #model used is changing here
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
class_names = list(validation_generator.class_indices.keys())



plot_confusion_matrix(cm, class_names)

# In[61]:
#Uses the functions given in [41] and plot the history graphs and confusion matrix
# Plot confusion matrix
validation_generator.reset()
y_true = validation_generator.classes
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
class_names = list(validation_generator.class_indices.keys())


plot_confusion_matrix(cm, class_names)
# In[63]:
acc += history_2.history['accuracy']
val_acc += history_2.history['val_accuracy']

loss += history_2.history['loss']
val_loss += history_2.history['val_loss']

plot_history(acc,val_acc,loss,val_loss,3,4)
'''


