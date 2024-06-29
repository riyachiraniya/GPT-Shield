#import necessary libraries

#for building interactive web applications
import streamlit as st 

#to load the GPT-2 model and tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch
import nltk

#for breaking text into tokens
nltk.download('punkt')
nltk.download('stopwords')
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_sequence
from nltk.probability import FreqDist
import plotly.express as ps
from collections import Counter
from nltk.corpus import stopwords
import string

#Load gpt2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

#Computes the perplexity score of a text using the loaded GPT-2 model
def calculate_perplexity(text):
    #tokenize, encode and return tensors
    encoded_input = tokenizer.encode(text, add_special_tokens= False, return_tensors = 'pt')
    #extract the tensor for the single input
    input_ids = encoded_input[0]

    #disables gradient calculation
    with torch.no_grad():
        #calculate output
        outputs  = model(input_ids)
        #logits are raw output scores(before applying softmax function)
        logits = outputs.logits

    # Compute the loss using CrossEntropyLoss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, logits.size(-1)), input_ids.view(-1))

    # Compute perplexity
    #to measure of how well a model predicts a sample(lower is better)
    perplexity = torch.exp(loss)
    #to return perplexity score as scalar value
    return perplexity.item()


#Calculates the burstiness score of a text based on the frequency of repeated words
def calculate_burstiness(text):
    tokens = nltk.word_tokenize(text.lower())
    word_freq = FreqDist(tokens)
    repeated_count = sum(count>1 for count in word_freq.values())
    #provides a measure of how concentrated or clustered repeated words are in the text
    burstiness_score = repeated_count / len(word_freq)
    return burstiness_score


#Generates a plot of the top 10 most repeated words in the text
def  plot_top_repeated_word(text):
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]

    #creates a dictionary-like object where keys are tokens and values are their respective counts
    word_count = Counter(tokens)
    top_words = word_count.most_common(10)

    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]

    #plot
    fig = ps.bar(x= words, y = counts, labels = {'x': 'words', 'y':'counts'}, title = "Top 10 most repeated words")
    st.plotly_chart(fig, use_container_width = True)


#for creating a streamlit application
st.set_page_config(layout = "wide")
#for full width layout
st.title("Plagiarazzi")
st.subheader("AI Plagiarism Detector")
text_area = st.text_area("Enter your text")
if text_area is not None:
    if st.button("Analyze"):
        #to divide the text window into 3 columns of equal width
        col1, col2, col3 = st.columns([1,1,1])
        
        with col1:
            st.info("Your Input Text")
            st.success(text_area)
       
        with col2:
            st.info("Calculated Score")
            perplexity = calculate_perplexity(text_area)
            burstiness_score = calculate_burstiness(text_area)

            st.success("Perplexity Score: " +str(perplexity))
            st.success("Burstiness Score: " +str(burstiness_score))

            if perplexity > 22000 and perplexity < 100000 and burstiness_score <= 0.25:
                st.error("Text Analysis Result: AI Generated Content")

            else:
                st.success("Text  Analysis Result: Likely not generated with AI")

            st.warning("Disclaimer: Plagiarazzi provides an estimation of originality, but is not a foolproof method. Human review and verification are necessary to confirm results."
                       "By using it, you acknowledge that you have read and understood these disclaimers and agree to use the tool responsibly.")

        with col3:
            st.info("Basic Insights")
            plot_top_repeated_word(text_area)



