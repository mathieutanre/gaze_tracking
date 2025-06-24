import pickle

# Remplacez 'votre_fichier.pickle' par le chemin vers votre fichier pickle
fichier_pickle = 'scores.pckl'

# Ouvrir le fichier pickle en mode lecture binaire
with open(fichier_pickle, 'rb') as fichier:
    contenu = pickle.load(fichier)

# Afficher le contenu du fichier pickle
print(contenu)
