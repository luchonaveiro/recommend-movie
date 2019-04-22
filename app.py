import pandas as pd
import numpy as np
from keras.models import load_model
import flask

# Defino el movie index
data = pd.read_csv('data/imdb5000.csv')
movies = pd.unique(data['movie_title'])
movie_index = {movie: idx for idx, movie in enumerate(movies)}
index_movie = {idx: movie for movie, idx in movie_index.items()}

# Cargo el modelo
model = load_model('model/first_embedding_movie_recommendation.h5')

# Extraigo los embeddings y los normalizo (para poder hacer cosine similarity)
movie_layer = model.get_layer('movie_embedding')
movie_weights = movie_layer.get_weights()[0]
movie_weights = movie_weights / np.linalg.norm(movie_weights, axis = 1).reshape((-1, 1))

# Inicio la aplicacion
app = flask.Flask(__name__)

@app.route("/movie_list")
def show_movies():
    json_movies = data[['movie_title','director_name','actor_1_name','actor_2_name','actor_3_name','title_year','genres']].to_json(orient='records')
    resp = flask.Response(json_movies, status=200, mimetype='application/json')

    return resp


# Defino el endpoint que recomiendo peliculas
@app.route("/recommend_movies/<name>")
def find_similar(name, weights = movie_weights, index_name = 'movie', n = 11, least = False):
    
    # Select index and reverse index
    if index_name == 'movie':
        index = movie_index
        rindex = index_movie
    #elif index_name == 'feature':
    #    index = feature_index
    #   rindex = index_feature
    
    # Check to make sure `name` is in index
    try:
        # Calculate dot product between book and all others
        dists = np.dot(weights, weights[index[name]])
    
        # Sort distance indexes from smallest to largest
        sorted_dists = np.argsort(dists)
        
        # If specified, find the least similar
        if least:
            # Take the first n from sorted distances
            closest = sorted_dists[:n]
            
        # Otherwise find the most similar
        else:
            # Take the last n sorted distances
            closest = sorted_dists[-n:]
            closest = reversed(closest)
        
        df_recommendation = pd.DataFrame(closest)
        df_recommendation['movie_title'] = ''
        df_recommendation['score'] = 0.0

        for i in range(0,len(df_recommendation)):
            df_recommendation['movie_title'][i] = rindex[df_recommendation.iloc[i,0]]
            df_recommendation['score'][i] = dists[df_recommendation.iloc[i,0]]
        
        df_recommendation = df_recommendation[['movie_title','score']]
        df_recommendation = df_recommendation.iloc[1:11,]

        json_recomendacion = df_recommendation.to_json(orient='records')

        resp = flask.Response(json_recomendacion, status=200, mimetype='application/json')
    
    except:
        resp = []


        # # Print the most similar and distances
        # for c in reversed(closest):
        #     print(f'{index_name.capitalize()}: {rindex[c]:{80}} Similarity: {dists[c]:.{2}}')

    return resp


if __name__ == '__main__':
    app.run(debug=True)


# Ejemplos local
# Listado de peliculas: http://127.0.0.1:5000/movie_list
# Recomendacion de peliculas: http://127.0.0.1:5000/recommend_movie/Titanic
