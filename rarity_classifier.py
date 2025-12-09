import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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

    IMG_SIZE = (626, 457) #all (hopefully) mtg images are this size
    BATCH_SIZE = 32
    CLASSES = ["common", "uncommon", "rare", "mythic"]
    INT_CLASSES = [0, 1, 2, 3]
    EPOCHS = 5

    df = pd.read_csv("commander-cards-filtered.csv")

    df = df[(df['rarity'].isin(['rare', 'mythic']))] #selects only rare & mythic cards

    #Assign data to cohorts
    X = df['name'].values
    y = df['rarity'].values

    X_train, y_train, X_test, y_test, X_val, y_val = split_data(X,y)

    #Make datasets 
    train_ds = image_dataset_from_directory(
        "R&M Images/Train",
        seed=12345,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels=y_train,
        label_mode='int')

    test_ds = image_dataset_from_directory(
        "R&M Images/Test",
        seed=12345,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels=y_test,
        label_mode='int')

    val_ds = image_dataset_from_directory(
        "R&M Images/Val",
        seed=12345,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels=y_val,
        label_mode='int')
    
    # Model
    model = EfficientNetB0(weights="imagenet", include_top = True, input_shape = (IMG_SIZE, 3))

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
        verbose = 1
    )

    testSetPredictions = model.predict(test_ds)

    threshold = 0.5
    testSetPredictions = np.where(testSetPredictions > threshold, 1,0)

    testSetPredClasses = np.apply_along_axis(np.argmax, 1, testSetPredictions)

    t_classReportV2 = classification_report(y_test, testSetPredClasses, target_names=CLASSES, output_dict = True)
    t_classReportDF = pd.DataFrame(t_classReportV2).transpose().round(2)
    t_classReportDF.to_html("testClassReport.html")

    t_cm = confusion_matrix(y_test, testSetPredClasses, labels = INT_CLASSES)
    t_cm = pd.DataFrame(t_cm, columns = ['Predicted MEL', 'Predicted Other'], index = ['Actual MEL', 'Actual Other'])
    t_cm.to_html("testConfusionMatrix.html")







if __name__ == "__main__": 
    main()