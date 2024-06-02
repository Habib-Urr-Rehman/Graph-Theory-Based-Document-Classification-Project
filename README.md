# Document Classification Project

## Project Overview

This project is about classifying documents into three categories: Business & Finance, Diseases & Symptoms, and Fashion & Beauty. The primary goal is to demonstrate how machine learning can be used for text classification using K-Nearest Neighbors (KNN) and TF-IDF (Term Frequency-Inverse Document Frequency) features.

## Data Collection

1. **Web Scraping**: Data was scraped from 15 different websites for each of the three categories:
   - 15 documents related to Business & Finance
   - 15 documents related to Diseases & Symptoms
   - 15 documents related to Fashion & Beauty

2. **Data Division**: 
   - **Training Data**: 12 documents from each category (12 * 3 = 36 documents)
   - **Testing Data**: 3 documents from each category (3 * 3 = 9 documents)

## Approach

1. **File Reading**: 
   - Function to read text files from a directory and store the content and filenames in a DataFrame.

2. **TF-IDF Vectorization**:
   - Transform the text data into TF-IDF features using `TfidfVectorizer`.

3. **Data Splitting**:
   - Split the combined data into training and test sets.

4. **KNN Classification**:
   - Use KNN classifier to fit the training data and make predictions on the test data.

5. **Evaluation**:
   - Calculate the accuracy of the model.
   - Generate and display the confusion matrix.

6. **New Data Prediction**:
   - Predict the categories for new test documents.

## How to Run the Project

### Directory Structure
- **Training Folders**: 
  - `Business & Finance`: Contains 12 text files.
  - `Diseases & Symptoms`: Contains 12 text files.
  - `Fashion & Beauty`: Contains 12 text files.
- **Testing Folder**: 
  - `test data`: Contains 9 text files (3 from each category).

### Steps to Run
1. Ensure you have the necessary packages installed:
2. Execute the code block starting with `######### Apply KNN ###############`.

## Code Explanation

### Reading Files from Directory
The function `read_files_from_directory(directory)` reads all text files from the given directory and returns their content in a DataFrame.

### Combining Data
Data from all categories are combined into a single DataFrame, and labels are assigned accordingly.

### TF-IDF Vectorization
The text data is transformed into numerical features using TF-IDF, which helps in representing the importance of words in the documents.

### Splitting Data
Data is split into training (80%) and testing (20%) sets.

### KNN Classifier
The KNN classifier is initialized with `k=5`, trained on the training data, and used to make predictions on the test data.

### Model Evaluation
Accuracy is calculated, and a confusion matrix is generated to visualize the performance of the classifier.

### Predicting on New Data
The trained model is used to predict the categories of new test documents, and the results are displayed.

### Confusion Matrix Visualization
A heatmap is created to show the confusion matrix, providing a visual representation of the classifier's performance.

## Example Usage
To use the project, follow the code and instructions provided in the `main.py` file. The project should run successfully with the provided data and setup.

This README provides a comprehensive overview of the project, explaining each step in simple terms. By following the instructions, you should be able to run the document classification project and understand its workings.
