import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

st.title("Book Recommendation System")

df = pd.read_csv('data/books.csv', on_bad_lines='skip')

df2 = df.copy()

df2.loc[(df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
df2.loc[(df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
df2.loc[(df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
df2.loc[(df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
df2.loc[(df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"

rating_df = pd.get_dummies(df2['rating_between'])
language_df = pd.get_dummies(df2['language_code'])

features = pd.concat([rating_df, language_df, df2['average_rating'], df2['ratings_count']], axis=1)

min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)

model = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(features)
dist, idlist = model.kneighbors(features)

def BookRecommender(book_name):
    book_list_name = []
    book_id = df2[df2['title'] == book_name].index
    if len(book_id) > 0:
        book_id = book_id[0]
        for newid in idlist[book_id]:
            book_list_name.append(df2.loc[newid].title)
    return book_list_name

book_name = st.selectbox('Choose a book for recommendations', df2['title'].unique())

if st.button('Show Recommendations'):
    recommendations = BookRecommender(book_name)
    st.write("Recommended Books:")
    for book in recommendations:
        st.write(book)
