import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
# os.environ["TF_ENABLE_XLA"] = "0"  
import tensorflow as tf
#print(tf.__version__)
# tf.config.optimizer.set_jit(False)
#from tensorflow.keras.applications import EfficientNetB0
#from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import sys

#BASH THESE TO CREATE CONDA ENVIRONMENT WITH CORRECT VERSIONS (if you want gpu (you want gpu))
# conda create -n tf-gpu python=3.10 -y
# conda activate tf-gpu
# pip3 install tensorflow==2.15.*
# conda install -c nvidia cuda-cudart=12.2 cuda-nvcc=12.2 cuda-nvrtc=12.2 -y
# export CUDA_HOME=$CONDA_PREFIX
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# source ~/.bashrc
# conda activate tf-gpu
# pip3 install pandas
# pip3 install scikit-learn
# python3 -m pip install 'tensorflow[and-cuda]'

# Note: if you change this here, change it in save_images.py too
SAMPLE_SIZE = 1000


def main(): 
    args = sys.argv
    if len(args) != 2:
        sys.exit("Usage: python rarity_classifier.py [colors|rarity]")
    else:
        parameter = args[1]
    
    if parameter not in ["colors", "rarity"]:
        sys.exit("parameter must be 'colors' or 'rarity'")


    IMG_SIZE = (224,224) # resizes images
    BATCH_SIZE = 64
    EPOCHS = 40

    # read in dataset and sample randomly
    df = pd.read_csv("commander-cards-filtered.csv")
    df = df.sample(n=SAMPLE_SIZE, random_state=12345).reset_index(drop=True)

    #Assign data to cohorts
    X = df['name'].values
    try:
        y = df[parameter].values
    except Exception:
        sys.exit("parameter must be colors or rarity")

    #Make datasets 
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "Cards/Train",
        seed=12345,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "Cards/Test",
        seed=12345,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "Cards/Val",
        seed=12345,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    # forgot what this does exactly
    class_names = test_ds.class_names
    CLASSES = train_ds.class_names
    INT_CLASSES = list(range(len(CLASSES)))

    # also this
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
    ])

    # and this
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    # Create the model, start with imagenet weights and initially lock model
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False

    # create inputs and outputs
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)

    # build and compile model
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"]
    )

    # implement early stopping to avoid overfitting
    callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    ]

    # Train the model
    trainingHist = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # Now we unfreeze some layers to fine tune the model
    FINE_TUNE_AT = 200

    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    for layer in base_model.layers[FINE_TUNE_AT:]:
        layer.trainable = True

    # recompile the model with a lower learning rate and some unfrozen layers
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"]
    )

    fine_tune_epochs = 10

    # Continue training the model with some layers unfrozen
    history_2 = model.fit(
        train_ds,
        epochs=fine_tune_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    testSetPredictions = model.predict(test_ds)
    testSetPredClasses = np.argmax(testSetPredictions, axis=1)

    # get labels for classification report
    true_labels = np.concatenate([y for x, y in test_ds], axis=0)

    # Generate classification report and confusion matrix
    t_classReportV2 = classification_report(
        true_labels,
        testSetPredClasses,
        labels=INT_CLASSES,
        target_names=CLASSES,
        output_dict=True
    )
    t_classReportDF = pd.DataFrame(t_classReportV2).transpose().round(2)
    t_classReportDF.to_html("testClassReport.html")

    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    t_cm = confusion_matrix(y_true, testSetPredClasses)

    col_names = [f"Predicted {c}" for c in class_names]
    idx_names = [f"Actual {c}" for c in class_names]

    t_cm = pd.DataFrame(t_cm, columns=col_names, index=idx_names)
    t_cm.to_html("testConfusionMatrix.html")

if __name__ == "__main__":
    main()