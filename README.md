# cs373_MTG_Art

Names: Theo McGlashan

Things to do:

    Collect dataset. Dataset is going to be unusual compared to normal datasets. Instead of one label column and many attribute columns,
    there will be one attribute column and several label columns. The attribute column will be image data, likely represented by a 3d matrix of the form [x][y][rgb].
    The things that we want to predict are the other attributes of the cards. Let's say we want to predict:
        
        Color   Mana value  Release year    Text?? (some sort of text generation, maybe this predicts based off of the image and the other 3 labels)
    
        Idea: We find the label that we can predict with the highest accuracy just using image data. Then, we first predict that label, then use the predicted label and image data
        as attributes to predict the next label, again chosing the label that we can predict with the best accuracy. We do this for all 3 labels that aren't text, then use all
        3 labels and the image data as attributes to generate text.

        Idea: We split the data into pieces (perhaps fifths), and predict labels for one piece with the other pieces. We do this for each piece, so we have predictions for all cards
        that we store as a predictions dataset. This would allow a finished product to make instantaneous predictions for all cards without the need to retrain a model.

        Idea: In addition to the above thing, we also allow for the option of (hopefuly) slightly more accurate predictions with more runtime. This would work by using the data for
        every card besides the one we want to predict to predict labels for the one card. This would mean training a model and using it to predict during runtime, but would hopefuly
        result in slightly more accurate predictions.

