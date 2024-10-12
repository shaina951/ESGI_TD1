import os
import click
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from models import make_model
from data import make_dataset
from feature import make_features

@click.group()
def cli():
    pass

@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="Fichier de données d'entraînement")
@click.option("--model_dump_filename", default="models/randomforest.joblib", help="Fichier pour sauvegarder ou charger le modèle")
@click.option("--model_type", type=click.Choice(['logistic', 'random_forest'], case_sensitive=False), default='random_forest', help="Choisissez le type de modèle")


def train(input_filename, model_dump_filename, model_type):
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"Le fichier {input_filename} n'existe pas.")

    df = make_dataset(input_filename)
    titles, y = make_features(df)  

    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(titles) 
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier()
        
    #Entrainement
    model.fit(X_vectorized, y)

    joblib.dump(model, model_dump_filename)
    joblib.dump(vectorizer, model_dump_filename.replace('.joblib', '_vectorizer.joblib')) 
    
    
@click.command()
@click.option("--input_filename", default="../data/raw/test.csv", help="Fichier de données d'entraînement")
@click.option("--model_dump_filename", default="models/randomforest.joblib", help="Fichier pour charger le modèle")
@click.option("--output_filename", default="../data/processed/prediction.csv", help="Fichier de sortie pour les prédictions")


def predict(input_filename, model_dump_filename, output_filename):
    model = joblib.load(model_dump_filename)
    vectorizer = joblib.load(model_dump_filename.replace('.joblib', '_vectorizer.joblib'))
    
    df = make_dataset(input_filename)
    titles, _ = make_features(df) 
    #Vectorisation de str à vecteur
    X_vectorized = vectorizer.transform(titles)
    #Prediction
    df["predictions"] = model.predict(X_vectorized)
    #Sauvegarde dans data/processed le fichier csv
    df.to_csv(output_filename, index=False)
    


@click.command()
@click.option("--input_filename", default="../data/raw/train.csv", help="Fichier de données d'entraînement")
@click.option("--model_dump_filename", default="models/randomforest.joblib", help="Fichier pour charger le modèle")

def evaluate(input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    titles, y = make_features(df)  

    model = joblib.load(model_dump_filename)
    vectorizer = joblib.load(model_dump_filename.replace('.joblib', '_vectorizer.joblib'))

    #Vectorisation
    X_vectorized = vectorizer.transform(titles)  

    #Résultats
    scores = cross_val_score(model, X_vectorized, y, cv=5) 
    print("Scores de validation croisée :", scores)
    print(f"Accuracy: {scores.mean():.4f}")
    
    
@click.command()    
def evaluate_model(model, X, y):

    scores = cross_val_score(model, X, y, cv=5)  
    print("Scores de validation croisée :", scores)

cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
