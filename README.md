# Predicting early (imminent) ICU admission and Prolonged Stay using Clinical Notes
This repository contains code for creating different machine learning models to predict imminent ICU admission and prolong stay at the ICU using clinical notes only. The notes are part of the MIMIC-III database. The SQL script for extracting the data from the database can be found here.

Our procedure for building the model is as follows:
1. Extract the notes from the database with relevant conditions as detailed in the SQL script
2. Use [scispacy](https://allenai.github.io/scispacy/) to tokenize the notes
3. Use TF-IDF processing for numericalizing the text
4. Build logistic regression, random forest, and gradient boosting machine models

We also build a CNN model using a one-hot count vector of the notes. The code for each of the models can be found in the corresponding folders.
