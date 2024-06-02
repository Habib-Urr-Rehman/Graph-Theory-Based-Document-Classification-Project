

        #################################Scrapping and cleaning Code###################
# import requests
# from bs4 import BeautifulSoup
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import string
# import re
#
# def get_website_text(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     paragraphs = soup.find_all('p')
#     text = ' '.join(paragraph.get_text() for paragraph in paragraphs)
#     return text
#
# # Function to preprocess text
# def preprocess_text(text):
#     text = re.sub(r'\d+', '', text)
#     words = word_tokenize(text)
#     words = [word.lower() for word in words if word.isalnum()]
#     stop_words = set(stopwords.words('english'))
#     words = [word for word in words if word not in stop_words]
#     lemmatizer = WordNetLemmatizer()
#     words = [lemmatizer.lemmatize(word) for word in words]
#     preprocessed_text = ' '.join(words)
#     return preprocessed_text
#
#
# def limit_text(text, limit=500):
#     # Split text into words
#     words = text.split()
#     # Join the first 500 words
#     limited_text = ' '.join(words[:limit])
#     return limited_text
#
# # Main function
# def main():
#     url = "https://www.mayoclinic.org/diseases-conditions/infectious-diseases/symptoms-causes/syc-20351173"
#
#     # Get text content from the website
#     website_text = get_website_text(url)
#
#     preprocessed_text = preprocess_text(website_text)
#
#  # Limit the preprocessed text to 500 words
#     limited_text = limit_text(preprocessed_text)
#
#     with open("D:/ALL COURSES/6th semester/Graph Theory/GT project/preprocessed_text.txt", "w") as file:
#         file.write(limited_text)
#
# if __name__ == "__main__":
#     main()








                   #########Graph Creation###############3
#
# import networkx as nx
# import matplotlib.pyplot as plt
#
# # Function to read text file and create a directed graph
# def create_graph(file_path):
#     # Read the text file
#     with open(file_path, 'r') as file:
#         text = file.read()
#
#     # Split text into sentences
#     sentences = text.split('\n')
#
#     # Create a directed graph
#     G = nx.DiGraph()
#
#     # Iterate through each sentence
#     for sentence in sentences:
#         words = sentence.split()
#         for i in range(len(words)-1):
#             # Add nodes for each word
#             G.add_node(words[i])
#             G.add_node(words[i+1])
#             # Add edge between consecutive words
#             G.add_edge(words[i], words[i+1])
#
#     return G
#
# # File path
# file_path = r'D:\ALL COURSES\6th semester\Graph Theory\GT project\Business & Finance\1.txt'
#
#
# # Create directed graph
# graph = create_graph(file_path)
#
# # Draw the graph
# pos = nx.spring_layout(graph)
# nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=20)
# plt.show()


        #########Apply KNN ###############3




import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to read files from a directory

def read_files_from_directory(directory):
    files = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        with open(path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            files.append({'filename': filename, 'content': content})
    return pd.DataFrame(files)

# Reading data from Business & Finance folder
business_finance_path = r'D:\ALL COURSES\6th semester\Graph Theory\GT project\Business & Finance'
business_finance_data = read_files_from_directory(business_finance_path)
business_finance_data['label'] = 'Business & Finance'

# Reading data from Diseases/Symptoms folder
diseases_symptoms_path = r'D:\ALL COURSES\6th semester\Graph Theory\GT project\Diseases  Symptoms'
diseases_symptoms_data = read_files_from_directory(diseases_symptoms_path)
diseases_symptoms_data['label'] = 'Diseases/Symptoms'

# Reading data from Fashion & Beauty folder
fashion_beauty_path = r'D:\ALL COURSES\6th semester\Graph Theory\GT project\Fashion & Beauty'
fashion_beauty_data = read_files_from_directory(fashion_beauty_path)
fashion_beauty_data['label'] = 'Fashion & Beauty'

# Combining data from all folders
combined_data = pd.concat([business_finance_data, diseases_symptoms_data, fashion_beauty_data], ignore_index=True)

# Creating TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Transforming text data into TF-IDF features
X = tfidf_vectorizer.fit_transform(combined_data['content'])

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, combined_data['label'], test_size=0.2, random_state=42)

# Initializing KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Fitting the model
knn_classifier.fit(X_train, y_train)

# Predicting on test data
y_pred = knn_classifier.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predicting on new test data
test_data_path = r'D:\ALL COURSES\6th semester\Graph Theory\GT project\test data'
test_data = read_files_from_directory(test_data_path)
X_new = tfidf_vectorizer.transform(test_data['content'])
predictions = knn_classifier.predict(X_new)

print("\nPredictions for test data:")
for i in range(len(test_data)):
    print(f"{test_data['filename'][i]} : {predictions[i]}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Creating a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=knn_classifier.classes_, yticklabels=knn_classifier.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
#


