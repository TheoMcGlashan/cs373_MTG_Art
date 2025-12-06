import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm # this isnt necessary, just adds progress bars to for loops (for downloads)
import urllib.request

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
def move_images(X, y, cohort):
    for i in tqdm(range(len(X))):
        urllib.request.urlretrieve(X[i], os.path.join("R&M Images", cohort, str(y[i]) + '.png'))

def main():
    df = pd.read_csv("commander-cards-filtered.csv")

    #df = df[(df['rarity'].isin(['rare', 'mythic']))] #selects only rare & mythic cards

    # Make directories 
    os.makedirs("R&M Images", exist_ok = True)
    os.makedirs("R&M Images\Train", exist_ok = True)
    os.makedirs("R&M Images\Test", exist_ok = True)
    os.makedirs("R&M Images\Val", exist_ok = True)

    #Assign data to cohorts
    X = df['art_crop'].values
    y = df['name'].values

    X_train, y_train, X_test, y_test, X_val, y_val = split_data(X,y)

    # Downloads images to their folder
    move_images(X_train, y_train, 'Train')
    move_images(X_test, y_test, 'Test')
    move_images(X_val, y_val, 'Val')

if __name__ == "__main__":
    main()


