import re
import pytesseract
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configuramos las stopwords 
# Tanto en español e inglés
stop_words = set(stopwords.words('english')).union(set(stopwords.words('spanish')))

def clean_text(text): # Limpia el texto: minúsculas, quita stopwords y caracteres no alfabéticos.
    
    if not isinstance(text, str):
        return ""

    text = text.lower() # A minúsculas
    text = re.sub(r'[^a-z\s]', '', text) # Solo letras y espacios
    tokens = word_tokenize(text)

    # Quitar stopwords
    cleaned_tokens = [w for w in tokens if w not in stop_words]

    return " ".join(cleaned_tokens)