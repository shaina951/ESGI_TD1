import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Assurez-vous de télécharger les ressources NLTK si ce n'est pas déjà fait
nltk.download('stopwords')

# Initialiser le stemming et les stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('french'))  # Remplacez 'french' par la langue souhaitée

def preprocess_text(text):
    # Nettoyer le texte : enlever les caractères spéciaux et les chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Mettre en minuscules
    text = text.lower()
    # Enlever les stop words et appliquer le stemming
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

def make_features(df):
    # Appliquer le prétraitement sur les titres
    titles = df['video_name'].astype(str).str.replace('"', '', regex=False)
    titles = titles.apply(preprocess_text)

    # Extraire les étiquettes cibles
    y = df["is_comic"]

    return titles.tolist(), y  # Retourner les titres sous forme de liste de chaînes
