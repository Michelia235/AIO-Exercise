import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#Question 10
vi_data_df = pd.read_csv(r"C:\Users\Administrator\Desktop\AIO-Exercise\Module 2\week4 - Statistics\vi_text_retrieval.csv")
context = vi_data_df['text']
context = [doc.lower() for doc in context]
tfidf_vectorizer = TfidfVectorizer()
context_embedded = tfidf_vectorizer.fit_transform(context)
context_embedded.shape #check shape
print(context_embedded.toarray()[7][0])

#Question 11:
def tfidf_search(question, tfidf_vectorizer, top_d=5):
    # Lowercasing before encoding
    query_lower = question.lower()
    
    query_embedded = tfidf_vectorizer.transform([query_lower])
    
    cosine_scores = cosine_similarity(context_embedded, query_embedded).reshape((-1,))
    # Get top k cosine scores and their indices
    results = []
    for idx in cosine_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            'id': idx,
            'cosine_score': cosine_scores[idx]
        }
        results.append(doc_score)
    
    return results

# Example usage
question = vi_data_df.iloc[0]['question']
results = tfidf_search(question, tfidf_vectorizer, top_d=5)
print(results[0]['cosine_score'])

#Question 12:
def corr_search(question , tfidf_vectorizer , top_d =5):
    # Lowercasing before encoding
    query_lower = question.lower()
    query_embedded = tfidf_vectorizer.transform([query_lower])
    corr_scores = np.corrcoef(
                                query_embedded.toarray()[0],
                                context_embedded.toarray()
                            )
    corr_scores = corr_scores[0][1:]
    # Get top k correlation scores and their indices    
    results = []
    for idx in corr_scores.argsort()[-top_d:][::-1]:
        doc = {
        'id': idx,
        'corr_score': corr_scores[idx]
        }
        results.append(doc)

    return results


question = vi_data_df.iloc[0]['question']
results = corr_search(question, tfidf_vectorizer, top_d=5)
second_result_corr_score = results[1]['corr_score']
print(second_result_corr_score)


