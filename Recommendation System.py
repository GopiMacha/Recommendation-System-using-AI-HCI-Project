# Recommendation system was developed based on data of Google Play Store and text similarity. 
# The system offered the same type of suggestions and popular items in recommend_apps recommend_by_category recommend_popular and main functions.

# importing libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading dataset
df = pd.read_csv("E://googleplaystore.csv")

# removing duplicate rows
df = df.drop_duplicates()

# removing missing important values
df = df.dropna(subset=['App', 'Category', 'Genres'])

# converting rating to numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# filling missing ratings
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())

# filtering invalid ratings
df = df[(df['Rating'] >= 1) & (df['Rating'] <= 5)]

# converting reviews to numeric
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

# filling missing reviews
df['Reviews'] = df['Reviews'].fillna(0)

# combining category and genres
df['content'] = df['Category'] + " " + df['Genres']

# resetting index
df = df.reset_index(drop=True)

# creating tfidf matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# calculating cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# creating index mapping
indices = pd.Series(df.index, index=df['App']).drop_duplicates()

# creating recommendation function
def recommend_apps(app_name, top_n=5):

    # checking app availability
    if app_name not in indices:
        return "App not found in dataset."
    
    # getting index
    idx = indices[app_name]
    
    # flattening similarity scores
    sim_scores = list(enumerate(cosine_sim[idx].flatten()))
    
    # sorting similarity values
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # selecting top results
    sim_scores = sim_scores[1:top_n+1]
    
    # getting indices
    app_indices = [i[0] for i in sim_scores]
    
    # returning recommendations
    return df[['App', 'Category', 'Genres', 'Rating']].iloc[app_indices]


# creating category recommendation
def recommend_by_category(category, top_n=5):

    # filtering category
    filtered = df[df['Category'].str.upper() == category.upper()]
    
    # checking empty category
    if filtered.empty:
        return "Category not found."
    
    # sorting by rating and reviews
    return filtered.sort_values(
        by=['Rating', 'Reviews'],
        ascending=False
    )[["App", "Category", "Rating"]].head(top_n)


# creating popular recommendation
def recommend_popular(top_n=5):

    # sorting popular apps
    return df.sort_values(
        by=['Rating', 'Reviews'],
        ascending=False
    )[["App", "Category", "Rating"]].head(top_n)


# creating interface
def main():

    # printing title
    print(" AI Recommendation System HCI ")
    
    # starting loop
    while True:
        print("\nSelect Option:")
        print("1. Recommend similar apps")
        print("2. Recommend by category")
        print("3. Show popular apps")
        print("4. Exit")
        
        # reading choice
        choice = input("Enter choice 1 to 4: ")
        
        # selecting similar apps
        if choice == '1':
            app = input("Enter app name: ")
            print("\nRecommended Apps:")
            print(recommend_apps(app))
        
        # selecting category
        elif choice == '2':
            category = input("Enter category: ")
            print("\nTop Apps in Category:")
            print(recommend_by_category(category))
        
        # showing popular apps
        elif choice == '3':
            print("\nPopular Apps:")
            print(recommend_popular())
        
        # exiting program
        elif choice == '4':
            print("Exiting system")
            break
        
        # handling invalid input
        else:
            print("Invalid choice try again")


# running main function
if __name__ == "__main__":
    main()