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

#IMPORTANT: Make sure this split data function is the same when youre saving images
# The only difference should be the y (target) value you select for and pass into split_data
def split_data(X, y): 
      # 70 train 15 test 15 val
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, 
        train_size = 0.7, test_size = 0.3, random_state=12345) 

    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, 
        train_size = 0.5, test_size = 0.5, random_state=12345)
    
    return X_train, y_train, X_test, y_test, X_val, y_val

def main(): 

    args = sys.argv
    if len(args) != 2:
        sys.exit("only 2 arg: set name, parameter")
    else:
        # set_name = args[1]
        set_name = "Return to Ravnica"
        parameter = args[1]


    IMG_SIZE = (224,224) #resizes images
    BATCH_SIZE = 64
    if parameter == "colors":
        CLASSES = ["W", "U", "B", "R", "G", "M", "C"]
    elif parameter == "rarity":
        CLASSES = ["common", "uncommon", "rare", "mythic"]
    if parameter == "rarity":
        INT_CLASSES = [0, 1, 2, 3]
    elif parameter == "colors":
        INT_CLASSES = [0, 1, 2, 3, 4, 5, 6]
    EPOCHS = 40

    df = pd.read_csv("commander-cards-filtered.csv")

    df = df[df["set_name"] == set_name]



    #df = df[(df['rarity'].isin(['rare', 'mythic']))] #selects only rare & mythic cards

    #Assign data to cohorts
    X = df['name'].values
    try:
        y = df[parameter].values
    except Exception:
        sys.exit("parameter must be colors or rarity")

    X_train, y_train, X_test, y_test, X_val, y_val = split_data(X,y)

    if parameter == "rarity":
        label_map = {"common": 0, "uncommon": 1,"rare": 2, "mythic": 3}
    elif parameter == "colors":
        label_map = {"W": 0, "U": 1, "B": 2, "R": 3, "G": 4, "M": 5, "C": 6}

    y_train_int = [label_map[x] for x in y_train.tolist()]
    y_test_int = [label_map[x] for x in y_test.tolist()]
    y_val_int = [label_map[x] for x in y_val.tolist()]

    FOLDER_NAME = set_name.replace(" ", "_")

    #Make datasets 
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        FOLDER_NAME + "/Train",
        seed=12345,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels=y_train_int,
        label_mode='int')

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        FOLDER_NAME + "/Test",
        seed=12345,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels=y_test_int,
        label_mode='int')

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        FOLDER_NAME + "/Val",
        seed=12345,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels=y_val_int,
        label_mode='int')
    
    # class_weights = class_weight.compute_class_weight(
    #     class_weight='balanced',
    #     classes=np.unique(y_train_int),
    #     y=y_train_int)
    
    # Convert to dict
    # class_weights = {i: w for i, w in enumerate(class_weights)}
    
    # Model
    base_model = tf.keras.applications.EfficientNetB0(weights=None, include_top = False, input_shape = IMG_SIZE + (3,))

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.3)(x)  # regularization
    output = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(loss = "sparse_categorical_crossentropy",
        optimizer = 'adam',
        metrics = ['acc'])
    
    # modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(modelDir + config.longRunName + '_{epoch:02d}.keras', monitor='val_loss', verbose=1, save_best_only=config.save_best_only)
    # callbacklist = []
    # callbacklist.append(modelCheckpoint)
    
    trainingHist = model.fit(
        train_ds, 
        epochs = EPOCHS, 
        validation_data = val_ds, 
        verbose = 1, 
        # class_weight = class_weights
        )

    testSetPredictions = model.predict(test_ds)

    # threshold = 0.5
    # testSetPredictions = np.where(testSetPredictions > threshold, 1,0)

    # testSetPredClasses = np.apply_along_axis(np.argmax, 1, testSetPredictions)

    testSetPredClasses = np.argmax(testSetPredictions, axis=1)

    t_classReportV2 = classification_report(y_test_int, testSetPredClasses, target_names=CLASSES, output_dict = True)
    t_classReportDF = pd.DataFrame(t_classReportV2).transpose().round(2)
    t_classReportDF.to_html("testClassReport.html")

    t_cm = confusion_matrix(y_test_int, testSetPredClasses, labels = INT_CLASSES)
    t_cm = pd.DataFrame(t_cm, columns = ['Predicted common', 'Predicted uncommon', 'Predicted rare', 'Predicted mythic'], index = ['Actual common', 'Actual uncommon', 'Actual rare', 'Actual mythic'])
    t_cm.to_html("testConfusionMatrix.html")



if __name__ == "__main__": 
    main()