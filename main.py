from sklearn.neighbors import NearestNeighbors
from algos import get_movie, get_movie_ids
from data import create_data_matrix, select_movies, get_inputs


def recommend_movie(movies, final_dataset, knn, csr_data, movie_inputs=None):
    # Get user inputs
    if movie_inputs is None:
        movie_inputs = get_inputs()
    else:
        movie_inputs = movie_inputs

    # Find matching movie IDs
    matched_movies, unmatched_counter = get_movie_ids(movie_inputs, movies)
    num_selections = len(movie_inputs) - unmatched_counter

    if num_selections > 0:
        # Select movies from the matched results
        selected_movie_ids = select_movies(matched_movies, num_selections)
        # Get recommendations based on selected movie IDs
        recommendations = get_movie(selected_movie_ids, final_dataset, knn, csr_data, movies)
        print(type(recommendations))

        # Print the first 5 recommendations to the user
        if type(recommendations) == str:
            print("There was not enough data for movie you like")
        
        else:
            print(recommendations.head(5))

            # If there are more than 5 recommendations, ask the user if they want to reroll
            if len(recommendations) > 5:
                choice = input("Do you want to reroll? (y/n): ")
                if choice.lower() == 'y':
                    print("Rerolling...")
                    # Print the remaining recommendations (indices 6-10)
                    print(recommendations.iloc[5:10])

    else:
        print("No movies were matched. Process halted. Starting again")
        return recommend_movie()



def main():

    final_dataset, movies, csr_data = create_data_matrix()

    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(csr_data)

    recommend_movie(movies, final_dataset, knn, csr_data)

if __name__ == "__main__":
    main()



