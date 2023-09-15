import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize session state for sentiment history
if 'sentiment_history' not in st.session_state:
    st.session_state.sentiment_history = pd.DataFrame(columns=['Timestamp', 'Text', 'Sentiment', 'Polarity', 'Neutral', 'Positive', 'Negative'])

# Create a VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment for a given text
def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    sentiment_score = sentiment_scores['compound']
    return sentiment_score

# Define a function to display individual text analysis
def display_text_analysis(text):
    sentiment_scores = analyze_sentiment(text)
    sentiment_score = sentiment_scores  # The sentiment score is already a float

    # Extract individual scores
    polarity_score = 0.5 + (sentiment_score - 0.5)
    neutral_score = 1 - abs(sentiment_score - 0.5)
    positive_score = max(0, sentiment_score)
    negative_score = max(0, -sentiment_score)

    # Calculate star rating based on sentiment scores
    if sentiment_score > 0.8:
        star_rating = "⭐⭐⭐⭐⭐"
    elif sentiment_score > 0.6:
        star_rating = "⭐⭐⭐⭐"
    elif sentiment_score > 0.4:
        star_rating = "⭐⭐⭐"
    elif sentiment_score > 0.2:
        star_rating = "⭐⭐"
    else:
        star_rating = "⭐"

    # Create a new entry as a dictionary
    new_entry = {
        'Timestamp': pd.Timestamp.now(),
        'Text': text,
        'Sentiment': sentiment_score,
        'Polarity': polarity_score,
        'Neutral': neutral_score,
        'Positive': positive_score,
        'Negative': negative_score
    }

    # Create a new DataFrame for the new entry
    new_df = pd.DataFrame([new_entry])

    # Concatenate the new DataFrame with the existing sentiment history
    st.session_state.sentiment_history = pd.concat([st.session_state.sentiment_history, new_df], ignore_index=True)

    # Display sentiment analysis results
    st.write(f'Sentiment Score: {sentiment_score:.2f}')
    st.write(f'Polarity Score: {polarity_score:.2f}')
    st.write(f'Neutral Score: {neutral_score:.2f}')
    st.write(f'Positive Score: {positive_score:.2f}')
    st.write(f'Negative Score: {negative_score:.2f}')
    st.write(f'Star Rating: {star_rating}')

# Define a function to display sentiment graph
def display_sentiment_graph():
    # Create a grouped bar chart for sentiment scores
    plt.figure(figsize=(10, 6))
    scores_df = st.session_state.sentiment_history[['Polarity', 'Neutral', 'Positive', 'Negative']]
    scores_df.columns = ['Polarity', 'Neutral', 'Positive', 'Negative']
    scores_df.sum().plot(kind='bar', color=['purple', 'blue', 'green', 'red'])
    plt.title('Sentiment Scores')
    plt.xlabel('Score Type')
    plt.ylabel('Score')
    st.pyplot(plt)

# Define a function to upload and analyze a file
def analyze_uploaded_file(file):
    try:
        if file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # Read Excel file into a DataFrame
            df = pd.read_excel(file)
        else:
            # Read CSV file into a DataFrame with 'ISO-8859-1' encoding
            df = pd.read_csv(file, encoding='ISO-8859-1')

        # Choose a column for sentiment analysis
        selected_column = st.selectbox('Choose a column for sentiment analysis', df.columns)

        # Perform sentiment analysis and append the result to a new column
        df['Sentiment'] = df[selected_column].apply(analyze_sentiment)

        # Define thresholds for sentiment classification
        positive_threshold = 0.5  # Adjust as needed
        negative_threshold = -0.5  # Adjust as needed

        # Function to classify sentiment
        def classify_sentiment(score):
            if score >= positive_threshold:
                return "Positive"
            elif score <= negative_threshold:
                return "Negative"
            else:
                return "Neutral"

        # Apply sentiment classification
        df['Sentiment_Class'] = df['Sentiment'].apply(classify_sentiment)

        # Display the first 30 rows of the updated table
        st.write(df.head(30))

        # Download analyzed file
        def download_analyzed_file(df, file_name, file_format):
            output = io.BytesIO()
            if file_format == 'CSV':
                df.to_csv(output, index=False)
            elif file_format == 'Excel':
                df.to_excel(output, index=False, engine='xlsxwriter')
            output.seek(0)
            return output
        
        # Download format
        download_format = st.radio('Select download format:', ['CSV', 'Excel'])
        if st.button('Download Analyzed File'):
            analyzed_file = download_analyzed_file(df, 'analyzed_file', download_format)
            st.download_button(
                label=f'Download {download_format} File',
                data=analyzed_file,
                file_name=f'analyzed_file.{download_format.lower()}',
                mime=f'application/{download_format.lower()}',
            )

        # button to generate graph for the sentiment scores
        if st.button('Generate Graph'):
            
            # Create a grouped bar chart for sentiment scores
            plt.figure(figsize=(10, 6))
            df['Sentiment_Class'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
            plt.title('Sentiment Analysis Results')
            plt.xlabel('Sentiment Class')
            plt.ylabel('Count')
            st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Main Streamlit app
st.header('Sentiment Analysis On Product Reviews')

# User input text area
text = st.text_area('Enter text for sentiment analysis:')

# Button to perform sentiment analysis for individual text
if st.button('Analyze Text'):
    if text:
        display_text_analysis(text)

# Button to generate a graph for sentiment scores
if st.button('Generate Sentiment Scores Graph'):
    display_sentiment_graph()
    
        
# Upload file functionality
file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if file:
    try:
        if file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # Read Excel file into a DataFrame
            df = pd.read_excel(file)
            st.subheader('Opened File Content:')
            st.write(df)
        else:
            # Read CSV file into a DataFrame with 'ISO-8859-1' encoding
            df = pd.read_csv(file, encoding='ISO-8859-1')
            st.subheader('Uploaded File Content:')
            st.write(df)
            
        # Choose a column for sentiment analysis
        selected_column = st.selectbox('Choose a column for sentiment analysis', df.columns, placeholder= 'select column')
        # Check the data type of the selected column
        if df[selected_column].dtype != 'object':
            st.warning("Selected column contains non-text data. Please choose a different column.")
        else:   
        # Perform sentiment analysis and append the result to a new column
            df['Sentiment'] = df[selected_column].apply(analyze_sentiment)

        # Define thresholds for sentiment classification
        positive_threshold = 0.5  # Adjust as needed
        negative_threshold = -0.5  # Adjust as needed

        # Function to classify sentiment
        def classify_sentiment(score):
            if score >= positive_threshold:
                return "Positive"
            elif score <= negative_threshold:
                return "Negative"
            else:
                return "Neutral"

        # Apply sentiment classification
        df['Sentiment_Class'] = df['Sentiment'].apply(classify_sentiment)

        # Display the first 30 rows of the updated table
        st.write(df.head(30))

        # Add the ability to download the analyzed file
        def download_analyzed_file(df, file_name, file_format):
            output = io.BytesIO()
            if file_format == 'CSV':
                df.to_csv(output, index=False)
            elif file_format == 'Excel':
                df.to_excel(output, index=False, engine='xlsxwriter')
            output.seek(0)
            return output

        download_format = st.radio('Select download format:', ['CSV', 'Excel'])
        if st.button('Download Analyzed File'):
            analyzed_file = download_analyzed_file(df, 'analyzed_file', download_format)
            st.download_button(
                label=f'Download {download_format} File',
                data=analyzed_file,
                file_name=f'analyzed_file.{download_format.lower()}',
                mime=f'application/{download_format.lower()}',
            )

        # button to generate a graph for the sentiment scores
        if st.button('Generate Graph'):
            # Create a grouped bar chart for sentiment scores
            plt.figure(figsize=(10, 6))
            df['Sentiment_Class'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
            plt.title('Sentiment Analysis Results')
            plt.xlabel('Sentiment Class')
            plt.ylabel('Count')
            st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    
# Download sentiment history as CSV
if st.button('Download Sentiment History'):
    csv_data = st.session_state.sentiment_history.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="Download Sentiment History as CSV",
        data=csv_data,
        file_name='sentiment_history.csv',
        mime='text/csv'
    )

# Display sentiment history
st.subheader('Sentiment History')
st.dataframe(st.session_state.sentiment_history)
