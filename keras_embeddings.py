import pandas as pd
import numpy as np
import keras
import pandasql as qy
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
import random

# Preproceso la data
def preprocess_data():
    # Importo la data
    data = pd.read_csv('data/imdb5000.csv')

    # Elijo variables a usar en los embeddings
    variables = ['movie_title','director_name','actor_1_name','actor_2_name','actor_3_name','genres','content_rating','country']
    data = data[variables]

    data['movie_title'] = data['movie_title'].apply(lambda x: x.strip('\xc2\xa0'))
    
    for column in data.columns:
        if column == 'plot_keywords':
            data[column] = data[column].fillna(value='')
        else:
            data.loc[data[column].isnull(), [column]] = data.loc[data[column].isnull(), column].apply(lambda x: [])

    data['genres'] = data['genres'].apply(lambda x: x.split('|'))

    # Agrego columna que junta todos los features
    for i in range(0,len(data)):
        if len(data['director_name'][i]) == 0:
            pass
        else:
            data['genres'][i].append(data['director_name'][i])

        if len(data['actor_1_name'][i]) == 0:
            pass
        else:
            data['genres'][i].append(data['actor_1_name'][i])
 
        if len(data['actor_2_name'][i]) == 0:
            pass
        else:
            data['genres'][i].append(data['actor_2_name'][i])

        if len(data['actor_3_name'][i]) == 0:
            pass
        else:
            data['genres'][i].append(data['actor_3_name'][i])

        if len(data['content_rating'][i]) == 0:
            pass
        else:
            data['genres'][i].append(data['content_rating'][i])

        if len(data['country'][i]) == 0:
            pass
        else:
            data['genres'][i].append(data['country'][i])
    
    data = data[['movie_title','genres']]
    
    return data

data = preprocess_data()

# Armo un indice de las peliculas
movies = pd.unique(data['movie_title'])
movie_index = {movie: idx for idx, movie in enumerate(movies)}
index_movie = {idx: movie for movie, idx in movie_index.items()}

# Armo lista con los uniques features
total_features = []
for i in range(0,len(data)):
    total_features = total_features + data['genres'][i]

def uniq(total_list):
    unique_list = []
    for i in range(0,len(total_list)):
        if total_list[i] in unique_list:
            pass
        else:
            unique_list.append(total_list[i])
    
    return unique_list

unique_features = uniq(total_features)

# Armo un indice de los features
feature_index = {feature: idx for idx, feature in enumerate(unique_features)}
index_feature = {idx: feature for feature, idx in feature_index.items()}

# Armo pares entre peliculas y features
pairs = []

for i in range(0,len(data)):
    movie_pair = movie_index[data['movie_title'][i]]
    for j in range(0,len(data['genres'][i])):
        feature_pair = feature_index[data['genres'][i][j]]
        pairs.append((movie_pair,feature_pair))

# Elimino pares duplicados
pairs_set = set(pairs)

# Funcion que genera batches de datos (positivos y negatiovos)
random.seed(100)

def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0, classification = False):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (movie_id, feature_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (movie_id, feature_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            random_movie = random.randrange(len(movies))
            random_feature = random.randrange(len(unique_features))
            
            # Check to make sure this is not a positive example
            if (random_movie, random_feature) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_movie, random_feature, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'movie': batch[:, 0], 'feature': batch[:, 1]}, batch[:, 2]


# Armo el modelo de embeddings
def movie_embedding_model(embedding_size = 50, classification = False):
    """Model to embed books and wikilinks using the functional API.
       Trained to discern if a link is present in a article"""
    
    # Both inputs are 1-dimensional
    movie = Input(name = 'movie', shape = [1])
    feature = Input(name = 'feature', shape = [1])
    
    # Embedding the movie (shape will be (None, 1, 50))
    movie_embedding = Embedding(name = 'movie_embedding',
                               input_dim = len(movie_index),
                               output_dim = embedding_size)(movie)
    
    # Embedding the feature (shape will be (None, 1, 50))
    feature_embedding = Embedding(name = 'feature_embedding',
                               input_dim = len(feature_index),
                               output_dim = embedding_size)(feature)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([movie_embedding, feature_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [movie, feature], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [movie, feature], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model

# Instantiate model and show parameters
model = movie_embedding_model()
model.summary()


# Entreno el modelo
n_positive = 500

gen = generate_batch(pairs, n_positive, negative_ratio = 2)

h = model.fit_generator(gen, epochs = 15, 
                        steps_per_epoch = len(pairs) // n_positive,
                        verbose = 2)

# Guardo el modelo
model.save('model/first_embedding_movie_recommendation.h5')

# Extraigo los embeddings y los normalizo (para poder hacer cosine similarity)
movie_layer = model.get_layer('movie_embedding')
movie_weights = movie_layer.get_weights()[0]
movie_weights = movie_weights / np.linalg.norm(movie_weights, axis = 1).reshape((-1, 1))

# Creo funcion para encontrar peliculas parecidas
def find_similar(name, weights, index_name = 'movie', n = 10, least = False, return_dist = False):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""
    
    # Select index and reverse index
    if index_name == 'movie':
        index = movie_index
        rindex = index_movie
    elif index_name == 'feature':
        index = feature_index
        rindex = index_feature
    
    # Check to make sure `name` is in index
    try:
        # Calculate dot product between book and all others
        dists = np.dot(weights, weights[index[name]])
    except KeyError:
        print(f'{name} Not Found.')
        return
    
    # Sort distance indexes from smallest to largest
    sorted_dists = np.argsort(dists)
    
    # If specified, find the least similar
    if least:
        # Take the first n from sorted distances
        closest = sorted_dists[:n]
         
        print(f'{index_name.capitalize()}s furthest from {name}.\n')
        
    # Otherwise find the most similar
    else:
        # Take the last n sorted distances
        closest = sorted_dists[-n:]
        
        # Need distances later on
        if return_dist:
            return dists, closest
        
        print(f'{index_name.capitalize()}s closest to {name}.\n')
        
    # Need distances later on
    if return_dist:
        return dists, closest
    
    
    # Print the most similar and distances
    for c in reversed(closest):
        print(f'{index_name.capitalize()}: {rindex[c]:{80}} Similarity: {dists[c]:.{2}}')


# Me fijo que predice
find_similar('Titanic', movie_weights)