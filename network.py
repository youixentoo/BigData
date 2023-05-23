# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:52:16 2023

@author: Thijs Weenink

Main file for model definition and training/validation/testing
"""
try:
    from memory_profiler import profile
except ImportError:
    print("""
################################### Warning ###################################
Module 'memory_profiler' is not installed, this is not required to run the code 
aslong as @profile remains commented out.
###############################################################################
""")
import file_parser as fp
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import argmax, unique, asarray
from sklearn.metrics import RocCurveDisplay
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Main function calling all the other functions
@profile
def main():
    # Batch size as per iterchunks; DO NOT CHANGE
    batch_size = 32
    
    # Settings
    dataset_loc = 'C:/Users/User/Desktop/School/BigData/Dataset'
    img_height = 96 
    img_width = 96
    epochs = 2
    
    # Labels test data
    test_labels = fp.get_labels_h5_file(dataset_loc, "test")
    # Use the test labels to determine how many different classes there are
    classes = unique(test_labels)
    num_classes = len(classes)

    # Check if files exist, calls sys.exit() if not
    fp.check_if_exists(dataset_loc, "train")
    fp.check_if_exists(dataset_loc, "valid")
    fp.check_if_exists(dataset_loc, "test")
    
    # Define the datasets
    train_dataset = fp.get_dataset(dataset_loc, "train", batch_size, img_height, img_width)
    valid_dataset = fp.get_dataset(dataset_loc, "valid", batch_size, img_height, img_width)
    
    # Model training/testing
    model, history = model_train_val(train_dataset, valid_dataset, num_classes, img_height, img_width, epochs)
    
    # No point keeping these around anymore, frees up alot of ram incase the dataset is cached.
    del train_dataset
    del valid_dataset
    
    # Test dataset
    test_dataset_no_labels = tf.data.Dataset.from_generator(
      fp._load_single_file_tfDataset,
      (tf.uint8),
      (tf.TensorShape((batch_size, img_height, img_width, 3))),
      args=(dataset_loc, "test", "x")) 
    
    # Plot the accuracy and loss
    plot_TV_acc_loss(history, epochs)
    # Use the test dataset to test the model
    results = predict_on_model(model, test_dataset_no_labels, batch_size)
    
    # ROC curve and auc
    pred_scores(results, test_labels, classes)
  

# Prediction scores
def pred_scores(results, test_labels, classes):      
    # Get predicted classes   
    predicted_class = [classes[argmax(tf.nn.softmax(reso))] for reso in results]
    pred_class_np = asarray(predicted_class)
    
    # Testing accuracy
    same_values = sum([pred_class_np[i]==test_labels[i] for i in range(pred_class_np.size)])/pred_class_np.size
    print(f"Testing accuracy: {(same_values[0]*100):.2f}%")
        
    # Displays the ROC curve aswell as the auc score
    RocCurveDisplay.from_predictions(
        pred_class_np,
        test_labels.ravel(),
        name="Model",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="best")


# Define model and train/validate. Returns model and history
def model_train_val(train_ds, val_ds, num_classes, img_height, img_width, epochs):
    
    # Default for Flip is 'HORIZONTAL_AND_VERTICAL'
    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip(input_shape=(img_height,
                                      img_width,
                                      3)),
        layers.RandomContrast(0.1),
      ]
    )

    """ Here for easy copy-paste
    layers.Dropout(0.2),
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    """
    
    # Define model
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255), # tranforms 0-255 uint8 data to 0-1
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # Standard Adam, 1e-3, gives 0.5 training accuracy (no prediction)
    # Switched over to Adam(1e-5)
    ### Copy-paste optimizers ###
    # keras.optimizers.RMSprop
    # optimizer=keras.optimizers.Adam(1e-5)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.summary()
    
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    return model, history  


# Uses the model to predict the outcome
def predict_on_model(model, dataset, batch_size):
    predictions = model.predict(dataset, batch_size=batch_size)
    return predictions
 
    
# Plots the training/validation accuracy/loss  
def plot_TV_acc_loss(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()



if __name__ == "__main__":
    main()