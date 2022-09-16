# Flatiron Phase 4 Project - Twitter Sentiment Analysis
~Benjamin Bai, Alvaro Mendizabal, Charles Pan


## Project Overview

In this project, we have been tasked to analyze over 8,000 tweets and create a natural-language processing model that can predict whether a tweet is positive, negative, or neutral. Through our strict filtering system, we are able to target specific tweets that are geared towards a company or product and feed those tweets into the prediction model. We can also produce WordClouds that display the top words associated with positive or negative sentiment regarding the company or product. The company that we have chosen to analyze is Apple, and our primary objective is to create a workflow in which we would receive a dataset of tweets, run the dataset through the filtering process and output predictions on the sentiment of those tweets and output WordClouds for those.


### Approach

We will go attempt to achieve our primary objective in 3 steps.

1. Data Cleaning: We first have to run some data cleaning process since the raw tweet has a lot raw data that needs to be standardized to a set format. The tweets will undergo a string cleaning step that will lowercase everything, remove things such as URL's, HTML elements, periods, etc., and lemmatize the data then remove the stopwords. Afterwards, the initial data felt incomplete in the that the emotion_in_tweet_is_directed_at was not picking up all the companies; thus, we created another categorizer that assigns which company a tweet is associated with based off keywords located within the tweets. Our categorizer distinguished the tweets much better than the company association that was initially assigned from the raw data.

2. Word Cloud: We have a word cloud analyzer that filters the dataframe to the specific parameters such as both sentiment as well as select words within each tweet. Afterwards, an image will be produced with the top words shown from tweets of those parameters with unnecessary words cleaned out. The purpose of the Word Cloud is for companys to analyze what words are associated with positive sentiment and what words are associated with negative sentiment. From there, they could choose to bolster or tackle the issue in regards to the sentiment behind those words.

3. NLP Prediction Model - The purpose of our model is to predict whether a tweet has positive, negative, or neutral sentiment. The model is based off supervised learning and is built off of a dataset of 8,000 tweets. The model is very useful as the end results can be used for further data exploration such as seeing the split between the sentiment for the current company. How the data is used is mostly up to how the company wants to use it, but our recommendation is to just track the overall sentiment over time to ensure that the reputation of the company is on an upwards trend.


### The Data

This project uses the Brand and Product Emotions dataset, which can be found in  `judge-1377884607_tweet_product_company.csv` in the data folder in this assignment's GitHub repository. Below are the columns in the data set. 
There isn't  much columns however the tweet_text column will be split into many columns when fed into our prediction model. It is crucial that the data cleaning process is done correctly.

* `tweet_text` - Content of the Tweet
* `emotion_in_tweet_is_directed_at` - Target of the Tweet (Apple Products/Google Products/etc.)
* `is_there_an_emotion_directed_at_a_brand_or_product` - Positive/Negative/Neutral Emotion

### Data Cleaning Procedure

The data cleaning procedure was one of the key components for ensuring that the data fed into the model is prim and proper.
The following process_string played a large component in cleaning our data. 
As you can see, the process_string function removes html/url text from the dataframe, sets everything to lowercase, lemmatizes, and then removes stopwords to prevent bloating as the keywords for sentiments don't really need the stopwords. Afterwards, we also produced functions which sorted whether a tweet was associated with Apple or Google throught the apple_sorter and google_sorter. The overall sorter was useful in that it could be re-appropriated to tackle different words as the function mostly looks through the tweets and creates a new column based off whether the text had the searched string in it or not. The last function that we used was the generate_word_cloud function as it outputs the WordCloud which essentially shows the most used words associated with the set parameters. (Whether it be sentiment and/or targetted group)
Process_String Function
```def process_string(text):
    """This function returns a processed list of words from the given text
    
    This function removes html elements and urls using regular expression, then
    converts string to list of workds, them find the stem of words in the list of words and
    finally removes stopwords and punctuation marks from list of words.
    
    Args:
        text(string): The text from which html elements, urls, stopwords, punctuation are removed and lemmatized
        
    Returns:
        clean_text(string): A text formed after text preprocessing.
    """
    
    # Remove any urls from the text
    text = re.sub(r'https:\/\/.*[\r\n]*',
                  "",
                  str(text))
    
    # Remove any urls starting from www. in the text
    text = re.sub(r'www\.\w*\.\w\w\w',
                  "",
                  str(text))
    
    # Remove any html elements from the text
    text = re.sub(r"<[\w]*[\s]*/>",
                  "",
                  str(text))
    
    # Remove prediods  marks
    text = re.sub(r"[\.]*",
                  "",
                  str(text))
    
 
    # Initialize RegexpTokenizer
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    tokenizer = RegexpTokenizer(pattern)

    
    # Tokenize text
    text_tokens = tokenizer.tokenize(text.lower())
    
    lemmatizer  = WordNetLemmatizer()
    # Get english stopwords
    english_stopwords = stopwords.words("english")
    new_list = ["mention", "sxsw", 'link', 'rt', 'quot']
    english_stopwords.extend(new_list)
    
    cleaned_text_tokens = [] # A list to hold cleaned text tokens
    
    for word in text_tokens:
        if((word not in english_stopwords) and # Remove stopwords
            (word not in string.punctuation)): # Remove punctuation marks
                
                lemmas = lemmatizer.lemmatize(word) # Get lemma of the current word
                cleaned_text_tokens.append(lemmas) # Appened lemma word to list of cleaned list

    # Combine list into single string
    clean_text = " ".join(cleaned_text_tokens)
    
    return clean_text
```
Apple Sorter/Google Sorter Function
```# define functions that loop through keyword lists and assign a category in a new column in the df
def apple_sorter(x):
    for i in is_apple:
        if i.lower() in x.lower():
            return 'Apple'
        else:
            continue
        
def google_sorter(x):
    for i in is_google:
        if i.lower() in x.lower():
            return 'Google'
        else:
            continue
```
Word_Cloud Function
```# generating a wordcloud with a twitter mask for positive words only 
wordcloud = WordCloud(stopwords=english_stopwords,
                      collocations=False, 
                      mask=twitter_mask, 
                      background_color='white', 
                      width=1800,
                      height=1400, 
                      contour_color='green', 
                      contour_width=2)

wordcloud.generate(','.join(apple_stopped))

plt.figure(figsize=(14, 14), 
           facecolor=None)

plt.imshow(wordcloud, 
           interpolation='bilinear')

plt.title('Apple Tweet Cloud', 
          size=20)

plt.axis("off")
```
### Findings and Conclusion

Our prediction model performed fairly well and was built off on XYZ algorithm.
The following hyper-parameters that the model used were...
1.
2.
3.
The following scores that the model generated were...
Accuracy Score:
Recall Score:
Precision Score:
F1 Score:
Below is a confusion matrix which demonstrates the spread of the predictions.
[Input Confusion Matrix.png]
In terms of the Word Clouds that were generated afterwards, this is one sample for Apple.
Below are the WordClouds for Positive and Negative Sentiment. The shape was have chosen was a bird since the text is based off Twitter and should be overall appealing for the consumer.
[Input WordCloud Birds.pngs]


### Recommendations

Our recommendation for what should be done with the data is that our prediction model and word clouds can be used to track overall sentiment. The data that has to be fed is just new tweets and those could be put into a time-series. Our model can predict the sentiment to a X accuracy and that can be used to track the overall trend of how people are viewing the company or product. It is up to the user on how they want to tackle the sentiment whether it is on an up-trend or down-trend. The purpose of the WordCloud is to see which words are generally associated with the positive and negative sentiments. For example, if iPhone was seen in the WordCloud for negative sentiment; it would be highly recommended to analyze any issues ongoing with the iPhone and why people may be viewing the iPhone in an unfavorable manner. 
Essentially, we cannot provide specific recommendations to the client as our model just displays the sentiment and top words associated with each sentiment and it is up to the client on how they want to deal with their trending sentiments on their company or product.


### Wanting to contribute?

If you are interested in contributing to our study (Please don't):

Fork this repository\
Clone your forked repository\
Add your scripts\
Commit and push\
Create a pull request\
Star this repository\
Wait for pull request to merge\
Allow us to review your contribution\

Repository Structure

├── README.md <- The README for reviewers of this project\
├── 01-Data_Cleaning.ipynb <- Documentation of the analysis process in Jupyter notebook\
├── 02-EDA_and_Regression_Model <- The presentation in PDF\
├── Data <- Both sourced externally and generated from code\
└── image <- Storage for all images used\