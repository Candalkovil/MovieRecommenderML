# MovieRecommenderML

This directory contains 3 files. main.py, algos.py, data.py

main.py -
  - Main method that imports needed functions and outputs movie results.
  - This function essentially packages all methods together

algos.py - 
  - Contains method called get_movie_ids() which is the NLP used to find which movie user is inputting
  - This is important since most of the times the user will not input a title that is matched in the dataset
  - Therefore, this function acts as an intermediary to find titles that the user could have meant
  - For example: user input = 'spider man', actual movie title = 'Spider-Man (2002)'.
  - Or user input = "mission imposible 5", actual title = "Mission Impossible: Rogue Nation"

  - Contains method called get_movie() which uses KNN and collaborative filtering to similar movies to inputs
  - Uses a distance based approach to recommend movies

data.py - 
  - Contains methods to import .csv files and data wrangling and filtering
  - Contains methods to transform dataframe into sparse matrix to ease functionality
