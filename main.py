import pickle
from sklearn.preprocessing import StandardScaler

#valence, key, mode, danceability, tempo, energy, loudness, speechiness, acousticness, instumentalness, liveness
example_song = [[0,5, 6, 1, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5]]

scaler = StandardScaler()
example_song = scaler.fit_transform(example_song)


with open('Project1_ML/mlp_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

print("prediction:", loaded_model.predict(example_song))

