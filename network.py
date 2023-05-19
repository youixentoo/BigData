# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:52:16 2023

@author: Thijs Weenink

Next: https://www.tensorflow.org/tutorials/images/classification#overfitting


WARNING:tensorflow:5 out of the last 5 calls to <function Model.make_predict_function.<locals>.predict_function at 0x000001EDDCEE0F70> 
triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function 
repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), 
please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that 
relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to
https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function 
for  more details.
"""
try:
    from memory_profiler import profile
except ImportError:
    print("Module 'memory_profiler' is not installed, this is not required to run the code, aslong as @profile remains commented out.")

import file_parser as fp
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import argmax, unique, asarray, uint8
from numpy import max as npmax
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# @profile
def main():
    # Batch size as per iterchunks; DO NOT CHANGE
    batch_size = 32
    
    # Settings
    dataset_loc = 'C:/Users/User/Desktop/School/BigData/Dataset'
    img_height = 96 
    img_width = 96
    epochs = 5
    
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
    
    # No point keeping these around anymore, frees up alot of ram
    del train_dataset
    del valid_dataset
    
    # Test data
    test_dataset_no_labels = tf.data.Dataset.from_generator(
      fp._load_single_file_tfDataset,
      (tf.uint8),
      (tf.TensorShape((batch_size, img_height, img_width, 3))),
      args=(dataset_loc, "test", "x")) 
    
    # Plot the accuracy and loss
    plot_TV_acc_loss(history, epochs)
    # Use the test dataset to test the model
    results = predict_on_model(model, test_dataset_no_labels, batch_size)
    
    # ROC curve
    pred_scores(results, test_labels, classes)
    
    # score = tf.nn.softmax(predictions[0])
    # Put predictions into a single array for roc calculation
    # predictions = asarray([argmax(tf.nn.softmax(result)) for result in results], dtype=uint8)
    
    # aera_under_curve = roc_calc_plot(test_labels.ravel(), results)
    # print(aera_under_curve)
  

# Prediction scores
def pred_scores(results, test_labels, classes):
    predicted_class = []
    confidence = []
    for res in results:
        score = tf.nn.softmax(res)
        pred = classes[argmax(score)]
        # print(pred)
        predicted_class.append(pred)
        confidence.append(npmax(score))
        
    
    pred_class_np = asarray(predicted_class)
    # conf_np = asarray(confidence)
    # print("pred_scores:",pred_class_np)
    # print("conf scores:",conf_np)
    # print("labels:", test_labels)
    
    # conf_scores = asarray([npmax(tf.nn.softmax(res)) for res in results], dtype=uint8)
    
    fp_rate, tp_rate, thresholds = roc_curve(test_labels, pred_class_np)
    
    # Had issues with this before, don't want to rerun the script if it for some reason fails 
    try:
        auc_score = roc_auc_score(fp_rate, tp_rate)
    except Exception as exc: # As I can't remember the exception
        print(exc)
        auc_score = 0
    # RocCurveDisplay.from_predictions(
    #     pred_class_np,
    #     test_labels.ravel(),
    #     name="Model",
    #     color="darkorange",
    # )
    plt.plot(fp_rate, tp_rate, label=f"Model (AUC = {auc_score})", color="dodgerblue")
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="best")


# Define model and train/validate. Retuns model and history
def model_train_val(train_ds, val_ds, num_classes, img_height, img_width, epochs):
    
    # Lets try some different augments, these make more sense to me than rotation/stretching
    # Default for Flip is 'HORIZONTAL_AND_VERTICAL' - RandomContrast
    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip(input_shape=(img_height,
                                      img_width,
                                      3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )
    
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

    # Standard Adam, 1e-3 too, gives 0.5 training accuracy (no prediction)
    # Switched over to Adam(1e-5)
    # keras.optimizers.RMSprop
    # optimizer=keras.optimizers.Adam(1e-5)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.summary()
    
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    return model, history  


# Calculate ROC and AUC and plot it, returns AUC
def roc_calc_plot(labels, predicted_labels):
    fp_rate, tp_rate, thresholds = roc_curve(labels, predicted_labels)
    auc_score = roc_auc_score(fp_rate, tp_rate)
    plt.plot(fp_rate, tp_rate, label=f"Model (area: {auc_score:.3f}") 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    return auc_score


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