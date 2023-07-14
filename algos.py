from fuzzywuzzy import fuzz
import pandas as pd
from operator import itemgetter

def get_movie_ids(user_inputs, movies_df, max_results_per_title=4):
    '''
    get_movie_ids() uses simple NLP in order to find what movie
    the user wants to input. With this, the user can input 'spider man' and
    this function will find what movies the user could be talking about, like 'Spider-Man 2 (2004)' '''
    # Check if the input is a list
    if not isinstance(user_inputs, list):
        raise ValueError("Input must be provided as a list.")

    # Initialize an empty DataFrame to store the matched movieIds and titles
    matched_movies = pd.DataFrame(columns=['movieId', 'title', 'similarity'])
    unmatched_counter = 0

    # Preprocess each user input
    preprocessed_inputs = [input.lower() for input in user_inputs]

    # Iterate over each movie input
    for user_input in preprocessed_inputs:
        # Initialize a list to store the matches for the current input
        input_matches = []

        # Iterate over each movie in the dataset
        for index, row in movies_df.iterrows():
            movie_id = row['movieId']
            movie_title = row['title'].lower()  # Convert the movie title to lowercase

            # Calculate the similarity score between the user input and movie title using fuzzy matching
            similarity_score = fuzz.token_set_ratio(user_input, movie_title)

            # Check if the input title is less than 4 letters or the number of matched movies is less than max_results_per_title
            if len(user_input) < 4 or len(matched_movies) < max_results_per_title:
                # Calculate the length similarity score based on the absolute difference in lengths
                length_similarity_score = 100 - abs(len(user_input) - len(movie_title))

                # Calculate the combined similarity score as a weighted average
                combined_similarity_score = (similarity_score + length_similarity_score) / 2
            else:
                combined_similarity_score = similarity_score

            # If the combined similarity score is above a certain threshold, consider it a match
            if combined_similarity_score >= 70:  # Adjust the threshold as per your requirements
                input_matches.append({'movieId': movie_id, 'title': row['title'], 'similarity': combined_similarity_score})

        # Sort the matches by similarity score in descending order
        input_matches.sort(key=itemgetter('similarity'), reverse=True)

        # Check if no matches were found for the current input
        if not input_matches:
            print(f"No matches found for movie title: {user_input}")
            print("Carrying on without")
            unmatched_counter += 1

        # Add the top matches to the final results, respecting the maximum number per input
        matched_movies = pd.concat([matched_movies, pd.DataFrame(input_matches[:max_results_per_title])], ignore_index=True)

    return matched_movies, unmatched_counter


def get_movie(movie_ids, final_dataset, knn, csr_data, movies):
    '''
    Main ML algo of the program. Uses KNN approach with collaborative filtering.
    To find closest distance between input movies and other movies.
    Then recommends list of movies to user based on the distance value'''
    n_movies_to_recommend = 10  # Adjust the number of movies to recommend
    movie_recommendations = []

    dataless_labels = 0
    series_names = set()

    for movie_id in movie_ids:
        movie_idx = final_dataset[final_dataset['movieId'] == movie_id].index
        if len(movie_idx):
            movie_idx = movie_idx[0]
            distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)
            rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
            recommend_frame = []
            for val in rec_movie_indices:
                movie_idx = final_dataset.iloc[val[0]]['movieId']
                idx = movies[movies['movieId'] == movie_idx].index
                title = movies.iloc[idx]['title'].values[0]

                # Check if the beginning name of the title already exists in the recommendations
                if any(title.startswith(name) for name in series_names):
                    continue  # Skip this movie if a similar series name is already included

                series_names.add(title.split()[0])  # Add the beginning name of the title to the set

                recommend_frame.append({'Title': title, 'Distance': val[1]})
            movie_recommendations.extend(recommend_frame[:n_movies_to_recommend])
        else:
            dataless_labels += 1
            # print(f"Not enough Data for one of the given movies")

    data_error = "There is not enough data for the movies you provided"
    if dataless_labels == len(movie_ids):
        return data_error

    df = pd.DataFrame(movie_recommendations, index=range(1, len(movie_recommendations) + 1))
    df = df.sort_values(by='Distance', ascending=True).reset_index(drop=True)

    return df
