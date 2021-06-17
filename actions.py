import pickle
model = pickle.load(open('./finalized_model.sav', 'rb'))
vect = pickle.load(open('./finalized_vectorizer.sav', 'rb'))

