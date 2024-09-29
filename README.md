# Multiclass Text Classification with News Data using Support Vector Machine (SVM) and Random Forest 
This project focuses on building a Multiclass Text Classification Model using news data scraped from various digital media platforms. The goal is to classify news articles into one of four categories: politics, sports, business, and entertainment. The project involves web scraping, text preprocessing, text representation using two methods, and model development using Support Vector Machine (SVM) and Random Forest classifiers.

## Project Objectives
1. **Data Collection**
   - Scrape news articles from at least three different media outlets inclouding Antara News, CNN, and Media Indonesia.
   - Gather at least 100 news articles with categories including politics, sports, business, and entertainment.
   - Perform manual labeling of the collected news articles based on their category.
2. **Text Preprocessing:**
    - Perform standard preprocessing tasks such as:
        - **Text Cleansing:** Remove unwanted characters, punctuation, and special symbols.
        - **Tokenization:** Split text into individual words or tokens.
        - **Filtering:** Remove stopwords and unnecessary tokens.
        - Optionally perform **Stemming** or **Lemmatization** to normalize words to their root form.
3. **Text Representation:**
   - Use two different methods for text representation:
       1. **TF-IDF (Term Frequency-Inverse Document Frequency):** A method that evaluates the importance of a word in a document relative to the entire dataset.
       2. **Word Embeddings:** Convert words into vector representations using context-based methods, with vector size set to 50 and a minimum word frequency of 3.
4. **Model Development:**
    - Build and tune two machine learning models:
       - **Support Vector Machine (SVM):** Adjust at least two hyperparameters.
       - **Random Forest:** Tune at least two hyperparameters such as the number of trees and maximum depth.
        
5. **Model Comparison:**
    - Compare the performance of both models using different text representation methods (TF-IDF vs Word Embeddings).
    - Evaluate the models using accuracy, precision, recall, and F1-score on the test data.
      
## Dataset Description
- **Source:** News articles scraped from media outlets such as Antara News, CNN, and Media Indonesia.
- **Columns:**
    - URL: Link to the news article.
    - Media: The source of the news article (Antara News, CNN, and Media Indonesia).
    - Label: The category label (Politics, Sports, Business, and Entertainment).

## Project Workflow
1. **Data Collection:**

    Web scraping is used to gather news articles from media platforms. The dataset contains URLs and labels indicating the article's category.
    
2. **Text Preprocessing:**
    - Clean and tokenize the news article text.
    - Perform stopword removal, stemming, and lemmatization if necessary.
3. **Text Representation:**
    - Apply TF-IDF to capture word importance in the document.
    - Use Word Embeddings to capture semantic word relationships in a 50-dimensional space.
4. **Model Development:**

    Train SVM and Random Forest classifiers, adjusting key hyperparameters for optimal performance.

5. **Model Evaluation:**

   Test and compare the models on performance metrics such as accuracy, precision, recall, and F1-score.

## Requirements
- Python 3.x
- Libraries:
  - `BeautifulSoup4`
  - `Requests`
  - `Pandas`
  - `NumPy`
  - `Scikit-learn`
  - `NLTK`
  - `Gensim`
  - `Matplotlib`
  - `Seaborn`
 
## Results
**Metode 1 (TF-IDF):**
- SVM achieved the highest accuracy (95%) with strong precision and recall across all classes.
- Random Forest improved significantly after parameter tuning, increasing its accuracy from 80% to 90%. This demonstrates the importance of parameter optimization in achieving better performance.

**Metode 2 (Word Embeddings):**
- SVM performed poorly, with very low accuracy (25%) and extremely low precision/recall values, indicating that the chosen parameters and representation method were not effective.
- Random Forest performed better than SVM but still had relatively lower accuracy (60-65%) compared to Metode 1. However, tuning helped improve the precision and recall slightly.

## Conclusion
- **Best Model:** The SVM model using TF-IDF (Metode 1) achieved the highest accuracy (95%) and outperformed all other models in terms of precision and recall.
- **Random Forest** improved after tuning, especially in Metode 1, but could not surpass the performance of SVM.
- **Metode 2 (Word Embeddings)** showed poor performance, particularly with SVM, indicating that this text representation method may not be suitable for this specific dataset or requires further refinement.
