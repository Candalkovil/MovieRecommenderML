import pandas as pd
from scipy.sparse import csr_matrix

def create_data_matrix():
    '''
    Data Wrangling and Filtering then transforms dataframe into sparse matrix
    '''
    movie_scores_df = pd.read_csv("movie_score.csv")
    movies = pd.read_csv("movies.csv")

    final_dataset = movie_scores_df.pivot(index='movieId',columns='userId',values='rating')

    final_dataset.fillna(0,inplace=True)
    final_dataset.head()

    no_user_voted = movie_scores_df.groupby('movieId')['rating'].agg('count')
    no_movies_voted = movie_scores_df.groupby('userId')['rating'].agg('count')

    final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]

    final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]



    csr_data = csr_matrix(final_dataset.values)
    final_dataset.reset_index(inplace=True)
    return final_dataset, movies, csr_data


def get_inputs():
    '''
    Simple user input to list function
    '''
    movie_inputs = []
    max_inputs = 5

    #Appends input strings into movie_inputs
    while len(movie_inputs) < max_inputs:
        movie_name = input("Enter a movie you like: ")
        movie_inputs.append(movie_name)

        #Check is list exceeds max_inputs
        if len(movie_inputs) >= max_inputs:
            break

        choice = input("Do you want to enter another movie? (y/n): ")
        if choice.lower() != 'y':
            break

    return movie_inputs


def select_movies(matched_movies, num_inputs):
    '''
    select_movies() allows the user to clarify which movies they meant in the input
    '''
    num_selections = num_inputs
    selected_movies = []

    #Displays titles relating to input text list
    print("Please select the movies you meant:")
    for i, movie in matched_movies.iterrows():
        print(f"{i+1}. {movie['title']}")

    while len(selected_movies) < num_selections:
        selection = input(f"Select movie {len(selected_movies)+1} (enter the number): ")

        try:
            selection_index = int(selection) - 1

            if selection_index >= 0 and selection_index < len(matched_movies):
                selected_movie = matched_movies.loc[selection_index]
                selected_movies.append(selected_movie)
                print(f"Selected movie: {selected_movie['title']}")
            else:
                print("Invalid selection. Please try again.")

        except ValueError:
            print("Invalid input. Please enter a number.")

    selected_ids = [movie['movieId'] for movie in selected_movies]
    return selected_ids