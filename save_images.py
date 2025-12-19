import os 
import pandas as pd
from sklearn.model_selection import train_test_split
#from tqdm import tqdm # this isnt necessary, just adds progress bars to for loops (for downloads)
import urllib.request
import sys

# Note: if you change this here, change it in classifier.py too
SAMPLE_SIZE = 1000

#IMPORTANT: Make sure this split data function is the same when youre training a model
# The only difference should be the y (target) value you select for and pass into split_data
def split_data(X, y): 
      # 70 train 15 test 15 val
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, 
        train_size = 0.7, test_size = 0.3, random_state=12345) 

    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, 
        train_size = 0.5, test_size = 0.5, random_state=12345)
    
    return X_train, y_train, X_test, y_test, X_val, y_val

# Copy images to assigned folders
def move_images(X, y, cohort, root):
    for i in range(len(X)):
        label = str(y[i])
        url = X[i]

        if not isinstance(url, str) or not url.startswith("http"):
            print(f"Skipping invalid URL for label {label}: {url}")
            continue

        class_dir = os.path.join(root, cohort, label)
        os.makedirs(class_dir, exist_ok=True)

        filename = url.split("/")[-1].split("?")[0]
        filepath = os.path.join(class_dir, filename)

        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print("Could not download image")
            print(f"  Label: {label}")
            print(f"  URL:   {url}")
            print(f"  Error: {e}")

def main():
    args = sys.argv
    if len(args) != 2:
        sys.exit("Usage: python save_images.py [colors|rarity]")
    else:
        PARAMETER = args[1]
    
    if PARAMETER not in ["colors", "rarity"]:
        sys.exit("parameter must be 'colors' or 'rarity'")
    
    ROOT_DIR = "Cards"

    df = pd.read_csv("commander-cards-filtered.csv")

    df = df.sample(n=SAMPLE_SIZE, random_state=12345).reset_index(drop=True)

    # Make directories
    os.makedirs(ROOT_DIR, exist_ok=True)
    os.makedirs(ROOT_DIR + "/Train", exist_ok=True)
    os.makedirs(ROOT_DIR + "/Test", exist_ok=True)
    os.makedirs(ROOT_DIR + "/Val", exist_ok=True)

    #Assign data to cohorts
    X = df['art_crop'].values
    y = df[PARAMETER].values

    X_train, y_train, X_test, y_test, X_val, y_val = split_data(X,y)

    # Downloads images to their folder
    move_images(X_train, y_train, 'Train', ROOT_DIR)
    move_images(X_test, y_test, 'Test', ROOT_DIR)
    move_images(X_val, y_val, 'Val', ROOT_DIR)

if __name__ == "__main__":
    main()


