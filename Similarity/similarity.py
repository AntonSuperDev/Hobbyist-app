from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizer, BertModel
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool
from contextlib import asynccontextmanager
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from typing import List
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util

data = {}

class SimilarityRequest(BaseModel):
    products: List[str]
    search_word: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting up")
    data["ner_tokenizer"] = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    data["ner_model"] = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    data["embedding_tokenizer"] = BertTokenizer.from_pretrained("bert-base-uncased")
    data["embedding_model"] = BertModel.from_pretrained("bert-base-uncased")
    data["sbert_model"] = SentenceTransformer('all-mpnet-base-v2')
    yield

app = FastAPI(lifespan=lifespan)

def get_named_entities(text, ner_model, ner_tokenizer):
    nlp_ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
    ner_results = nlp_ner(text)
    named_entities = [result['word'] for result in ner_results]
    return named_entities

# Function to calculate Jaccard similarity
def calculate_jaccard_similarity(search_word, product, ner_model, ner_tokenizer):
    set2 = set(get_named_entities(product.title(), ner_model, ner_tokenizer))
    set1 = set(get_named_entities(search_word.title(), ner_model, ner_tokenizer))
    intersection = set1.intersection(set2)
    union = set1
    print(set1)
    print(set2)
    
    if not union:
        return 0.0
    
    similarity = len(intersection) / len(union)
    return similarity

def calculate_cosine_similarity_sbert(search_word, product_title, sbert_model):
    # Get SBERT embeddings for search word and product title
    search_embedding = sbert_model.encode(search_word, convert_to_tensor=True)
    product_embedding = sbert_model.encode(product_title, convert_to_tensor=True)

    # Compute cosine similarity
    similarity = util.pytorch_cos_sim(search_embedding, product_embedding)
    similarity_score = similarity.item()
    return similarity_score

def calculate_cosine_similarity_sklearn(search_word, product_title):
    vectorizer = CountVectorizer()
    search_vector = vectorizer.fit_transform([search_word])
    product_vector = vectorizer.transform([product_title])
    similarity = cosine_similarity(search_vector, product_vector)
    return similarity[0][0]

def calculate_cosine_similarity_bert(search_word, product_title, embedding_tokenizer, embedding_model):
    # Tokenize input texts and convert them to tensors
    inputs1 = embedding_tokenizer(search_word, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs2 = embedding_tokenizer(product_title, return_tensors='pt', max_length=512, truncation=True, padding=True)

    # Forward pass through BERT model to get embeddings

    outputs1 = embedding_model(**inputs1)
    outputs2 = embedding_model(**inputs2)

    # Extract embeddings from BERT outputs
    embeddings1 = outputs1.last_hidden_state.mean(dim=1)  # Mean pooling across tokens
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)

    # Calculate cosine similarity
    similarity = cosine_similarity(embeddings1, embeddings2)

    return similarity.item()  # Return similarity as a single value

def remove_punctuation(s):
    s = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', '', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s

def preprocess_title(title):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    title = remove_punctuation(title)
    title = title.split()
    title = [word for word in title if word.lower() not in stop_words]
    title = [ps.stem(word) for word in title]

    return title

def original_simliarity(matching_title, target_title):
    matching_title = preprocess_title(matching_title)
    target_title = preprocess_title(target_title)

    intersection = len(set(matching_title) & set(target_title))

    if len(matching_title) == 0:
        return 0
    return intersection / (len(matching_title) + len(target_title) - intersection)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/calculate-similarity")
async def calculate_similarity_endpoint(request: SimilarityRequest):
    search_word = request.search_word
    products = request.products
    print(search_word)
    num_processes = 1
    with Pool(processes=num_processes) as pool:
        # similarity_scores_sbert = pool.starmap(calculate_cosine_similarity_sbert, [(search_word, product,data["sbert_model"]) for product in products])
        # similarity_scores_jaccard = pool.starmap(calculate_jaccard_similarity, [(search_word, product, data["ner_model"], data["ner_tokenizer"]) for product in products])
        # similarity_scores_bert = pool.starmap(calculate_cosine_similarity_bert, [(search_word, product,data["embedding_tokenizer"],data["embedding_model"]) for product in products])
        similarity_scores_skleran = pool.starmap(calculate_cosine_similarity_sklearn, [(search_word, product) for product in products])
        # similarity_scores_original = pool.starmap(original_simliarity, [(search_word, product) for product in products])

    return {
        # "sbert": similarity_scores_sbert,
        # "bert": similarity_scores_bert,
        "data": similarity_scores_skleran,
        # "jaccard": similarity_scores_jaccard,
        # "original": similarity_scores_original,
    }