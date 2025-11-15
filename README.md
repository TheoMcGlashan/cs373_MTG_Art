# cs373_MTG_Art

Names: Theo McGlashan

Things to do:

    Collect dataset. Dataset is going to be unusual compared to normal datasets. Instead of one label column and many attribute columns,
    there will be one attribute column and several label columns. The attribute column will be image data, likely represented by a 3d matrix of the form [x][y][rgb].
    The things that we want to predict are the other attributes of the cards. Let's say we want to predict:
        
        Color   Mana value  Release year    Text?? (some sort of text generation, maybe this predicts based off of the image and the other 3 labels)
    
