import nltk
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer, WordNetLemmatizer 
from keras.preprocessing import sequence
import pickle

# Cleaning comment
def clean_comment(comment):
    p = """'!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'"""
    stop_words = stopwords.words('english')
    stop_words.remove('not')
    comment = [word.lower() for word in comment]
    comment = [''.join(w for w in word if w not in p) for word in comment]
    comment = [word for word in comment if word not in stop_words]
    return comment

# Stemming and Lemmantization
def stemming(comment):
    return [SnowballStemmer(language="english").stem(word) for word in comment]

def lemmatization(comment):
    return [WordNetLemmatizer.lemmatize(word) for word in comment]

def prep_comment(comment):
    regexp = "([a-zA-Z]+(?:’[a-z]+)?)"
    regex_tokenizer = RegexpTokenizer(regexp)
    comment = regex_tokenizer.tokenize(comment)
    comment = clean_comment(comment)
    # comment = stemming(comment) 
    return ' '.join(comment)

def load_tokenizer():
	with open('tokenizer.pickle', 'rb') as handle:
		pkl = pickle.load(handle)
	return pkl

def process(text):
    maxlen = 35
    tokenizer = load_tokenizer()
    nltk.download('stopwords')
    cleaned_comment = prep_comment(text)		
    word_seq_train = tokenizer.texts_to_sequences(cleaned_comment)		
    wst = []
    for w in word_seq_train:
        if len(wst) == maxlen:
            break
        if len(w) != 0:
            wst.append(*w)		
    word_seq_train = sequence.pad_sequences([wst], maxlen=maxlen, padding="post")		
    return word_seq_train