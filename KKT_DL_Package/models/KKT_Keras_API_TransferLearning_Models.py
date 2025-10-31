# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 19:27:07 2025

@author: THYAGHARAJAN
"""
import tensorflow as tf

# In[]

def KKT_MobileNetV2_API_Model_with_Image_type_Selector(IMG_SIZE=(160, 160), image_type='rgb'):
    """
    Use this for binary classification
    Create a Keras model with the option to choose between RGB or Grayscale image inputs.
    The KKT model (KKTV1_MobileNetV2) is used for training because it includes both the 
    frozen base model and the trainable custom layers.
    imagenet weights are preloaded
    Parameters:
    - IMG_SIZE: Tuple, image size (height, width)
    - image_type: 'rgb' for RGB images, 'grayscale' for grayscale images
    
    Returns:
    - base_model: The feature extractor (MobileNetV2)
    - KKTV1_MobileNetV2: The complete model
    """
    if image_type == 'rgb':
        IMG_SHAPE = IMG_SIZE + (3,)  # 3 channels for RGB images
    elif image_type == 'grayscale':
        IMG_SHAPE = IMG_SIZE + (1,)  # 1 channel for grayscale images
    else:
        raise ValueError("Invalid image_type. Use 'rgb' or 'grayscale'.")
    
    # Load MobileNetV2 without the top layer for feature extraction
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, 
                                                   include_top=False,  # Exclude the top classification layer
                                                   weights='imagenet')  # use pre-trained weights

    # Global Average Pooling Layer to reduce feature map dimensions
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # Fully connected Dense layer for prediction
    prediction_layer = tf.keras.layers.Dense(1)  # Adjust based on task (e.g., binary classification)

    # Define the model input
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1 if image_type == 'grayscale' else 3))  # 1 or 3 channels

    x = inputs  # Start with the input tensor
    if image_type == 'grayscale':
        x = tf.image.grayscale_to_rgb(x)  # Convert grayscale to RGB by duplicating channels (3 channels)
    
    # Apply MobileNetV2 preprocessing.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)    
    
    x = base_model(x, training=False)  # Pass through the base model (feature extraction)
    x = global_average_layer(x)  # Global average pooling to reduce the feature map
    x = tf.keras.layers.Dropout(0.2)(x)  # Add dropout to reduce overfitting
    outputs = prediction_layer(x)  # Predictions

    # Define the complete model
    KKTV1_MobileNetV2 = tf.keras.Model(inputs, outputs)

    # Return the base model (for feature extraction) and the complete model
    return base_model, KKTV1_MobileNetV2



# In[]:
    
def KKT_MobileNetV2_Multiclass_Model(no_of_classes, IMG_SIZE=(160, 160), image_type='rgb'):
    """
    Create a Keras model with the option to choose between RGB or Grayscale image inputs.
    The KKT model (KKTV1_MobileNetV2) is used for training because it includes both the 
    frozen base model and the trainable custom layers.
    imagenet weights are preloaded
    Parameters:
    - no_of_classes  is given to the dense layer
    - IMG_SIZE: Tuple, image size (height, width)
    - image_type: 'rgb' for RGB images, 'grayscale' for grayscale images
    
    Returns:
    - base_model: The feature extractor (MobileNetV2)
    - KKTV1_MobileNetV2: The complete model
    """
    if image_type == 'rgb':
        IMG_SHAPE = IMG_SIZE + (3,)  # 3 channels for RGB images
    elif image_type == 'grayscale':
        IMG_SHAPE = IMG_SIZE + (1,)  # 1 channel for grayscale images
    else:
        raise ValueError("Invalid image_type. Use 'rgb' or 'grayscale'.")
    
    # Load MobileNetV2 without the top layer for feature extraction
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, 
                                                   include_top=False,  # Exclude the top classification layer
                                                   weights='imagenet')  # use pre-trained weights

    # Global Average Pooling Layer to reduce feature map dimensions
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # Fully connected Dense layer for prediction
    #prediction_layer = tf.keras.layers.Dense(1)  # For Binary classification outputs a number including negative numbers
    prediction_layer = tf.keras.layers.Dense(no_of_classes, activation='softmax')  # Change to match n classes, with 'softmax' activation for multi-class classification
    #softmax activation is used for multi-class classification because it outputs a probability distribution across all classes.
    # Define the model input
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1 if image_type == 'grayscale' else 3))  # 1 or 3 channels

    x = inputs  # Start with the input tensor
    if image_type == 'grayscale':
        x = tf.image.grayscale_to_rgb(x)  # Convert grayscale to RGB by duplicating channels (3 channels)
    
    # Apply MobileNetV2 preprocessing.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)    
    
    x = base_model(x, training=False)  # Pass through the base model (feature extraction)
    x = global_average_layer(x)  # Global average pooling to reduce the feature map
    x = tf.keras.layers.Dropout(0.2)(x)  # Add dropout to reduce overfitting
    outputs = prediction_layer(x)  # Predictions
    #outputs is a tensor of shape (batch_size, no_of_classes)
    #each row (one image)is a probability distribution of the classes (of that image) i.e. each row sums to 1.
    
    # Define the complete model
    KKTV1_MobileNetV2 = tf.keras.Model(inputs, outputs)
    #This model has random weights in the new Dense layer. So this is not trained model. 
    #If you use with .predict()  it will do forward classification. But dense layer has meaningless data
    #So, this model should be trained using .compile() and .fit()

    # Return the base model (for feature extraction) and the complete model
    return base_model, KKTV1_MobileNetV2    



# In[]:

"""
#Tensor flow model. May be deleted
'''
The KKT model (KKTV1_MobileNetV2) is used for training because it includes both the 
frozen base model and the trainable custom layers.
'''
def Create_Keras_API_model(train_dataset, IMG_SIZE = (160, 160) ):
    IMG_SHAPE = IMG_SIZE + (3,)  # resolution of the input image the model expects,  3 indicates RGB
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,       #instantiates the MobileNetV2 model
                                                   include_top=False,
                                                   weights='imagenet')   #initialize with pre-trained weights
    
    image_batch, label_batch = next(iter(train_dataset))
    #This line retrieves a batch of images and their corresponding labels from the train_dataset. 
    feature_batch = base_model(image_batch)
    #passes the image_batch through the base_model, 
    #transforming the input images into a higher-dimensional feature representation.
    print(feature_batch.shape)
    #The shape will typically be (batch_size, height, width, channels)
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average =global_average_layer (feature_batch)
    print(feature_batch_average.shape)
    
    prediction_layer = tf.keras.layers.Dense(1)  
    #last layer for regression. produces a single value. Sinle neuron for binary classifcation. no activation function
    prediction_batch = prediction_layer(feature_batch_average) 
    # takes hte features uses Dense layer weights and biases to produce output
    print(prediction_batch.shape)
    
    inputs = tf.keras.Input(shape=(160, 160, 3))
    # input tensor is expected to have a shape of (batch_size, 160, 160, 3)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False) #passes input tensor x through the base model. used only for inference
    x = global_average_layer(x)
    # global_average_layer is an instance of the GlobalAveragePooling2D layer in TensorFlow,
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    KKTV1_MobileNetV2 = tf.keras.Model(inputs, outputs)
    #KKTV1_MobileNetV2.summary()
    return base_model, KKTV1_MobileNetV2
"""