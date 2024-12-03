
import os
import streamlit as st
import pandas as pd
import numpy as np
from gradio_client import Client
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import xml.etree.ElementTree as ET
import json

# Set Hugging Face API tokens
os.environ['HF_TOKEN'] = "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"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to generate answers using LLaMA 2 model
def generate_answer(context, question):
    client = Client("huggingface-projects/llama-2-7b-chat")
    message = 'Stick to the context provided to answer the questions: ' + context + '\n\n' + question
    result = client.predict(
        message=message,
        system_prompt="Hello!!",
        max_new_tokens=4096,
        temperature=0.6,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        api_name="/chat"
    )
    return result

# Function to create summary and generate labels for a dataset
def summarize_and_label(data, dataset_name):
    context = f"This is a dataset named {dataset_name}. It contains the following columns: {', '.join(data.columns)}."
    summary = generate_answer(context, "Provide a brief summary of this dataset.")
    labels = generate_answer(context, "Generate label tags for this dataset.")
    return summary, labels

def generate_summary(data):
    summary = []
    
    # General information
    summary.append(f'This dataset contains {data.shape[0]} rows and {data.shape[1]} columns.')

    # Column descriptions
    for column in data.columns:
        summary.append(f"\nColumn '{column}':")
        
        # Data type and missing values
        dtype = data[column].dtype
        missing = data[column].isnull().sum()
        summary.append(f' - Data type: {dtype}')
        summary.append(f' - Missing values: {missing} ({(missing / data.shape[0]) * 100:.2f}%)')
        
        # Numerical column statistics
        if np.issubdtype(dtype, np.number):
            summary.append(f' - Mean: {data[column].mean():.2f}')
            summary.append(f' - Standard Deviation: {data[column].std():.2f}')
            summary.append(f' - Minimum: {data[column].min()}')
            summary.append(f' - Maximum: {data[column].max()}')
        
        # Categorical column statistics
        elif dtype == 'object':
            top_values = data[column].value_counts().head(3)
            summary.append(f' - Top values: {top_values.to_dict()}')
            
            # Generating a word frequency distribution for text columns
            text_data = data[column].dropna().str.cat(sep=' ')
            tokens = word_tokenize(text_data)
            filtered_tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
            freq_dist = FreqDist(filtered_tokens)
            common_words = freq_dist.most_common(3)
            summary.append(f' - Common words: {common_words}')
    
    return '\n'.join(summary)

# Plot distributions of numerical columns
def plot_distributions(df, filename='distributions.png'):
    plt.figure(figsize=(12, 8))
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        sns.histplot(df[column], kde=True, label=column)
    
    plt.title('Distribution of Numeric Columns')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Generate summary statistics for numerical columns
def get_summary_stats(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    return numeric_df.describe()

# Generate correlation matrix for numerical columns
def get_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    return numeric_df.corr()

# Outlier detection using IQR method
def detect_outliers(data):
    """Detects outliers in the dataset based on the IQR method."""
    numeric_data = data.select_dtypes(include=[np.number])  # Only select numeric data
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    is_outlier = (numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))
    return is_outlier

def remove_outliers(data):
    """Removes outliers from the dataset."""
    is_outlier = detect_outliers(data)
    # Ensure we only remove outliers from numeric data
    numeric_data = data.select_dtypes(include=[np.number])
    cleaned_data = numeric_data[~is_outlier.any(axis=1)]
    # Return original data with non-numeric columns intact
    return data.loc[cleaned_data.index]

def show_results(data, remove_outliers_option=False):
    """Displays the EDA results with or without outliers based on user input."""
    if remove_outliers_option:
        data = remove_outliers(data)
        st.write("### Outliers removed.")
    else:
        st.write("### Including outliers.")
    
    st.write("### Summary Statistics")
    st.write(get_summary_stats(data).to_html(classes="table table-striped"), unsafe_allow_html=True)
    st.write("### Correlation Matrix")
    st.write(get_correlation_matrix(data).to_html(classes="table table-striped"), unsafe_allow_html=True)
    st.write("### Data Distributions")
    plot_distributions(data)
    st.image('distributions.png')
    st.write("### Missing Values Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    fig = plt.gcf()  # Get the current figure
    st.pyplot(fig)

def load_data(file):
    file_type = file.name.split('.')[-1].lower()
    
    if file_type == 'csv':
        return pd.read_csv(file)
    elif file_type in ['xls', 'xlsx']:
        return pd.read_excel(file)
    elif file_type == 'json':
        return pd.read_json(file)
    elif file_type == 'xml':
        return pd.DataFrame(parse_xml(file))
    else:
        raise ValueError("Unsupported file format")

def parse_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    data = []

    for child in root:
        record = {}
        for subchild in child:
            record[subchild.tag] = subchild.text
        data.append(record)

    return data

def prepare_data_for_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Impute missing values for X
    imputer_X = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer_X.fit_transform(X.select_dtypes(include=[np.number])), columns=X.select_dtypes(include=[np.number]).columns)

    # Handle categorical features in X
    X_categorical = X.select_dtypes(include=['object'])
    for column in X_categorical.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X_categorical[column])

    # Impute missing values for y
    imputer_y = SimpleImputer(strategy='most_frequent' if y.dtype == 'object' else 'mean')
    y = pd.Series(imputer_y.fit_transform(y.values.reshape(-1, 1)).flatten(), name=target_column)

    return X, y

def summarize_image(image):
    # Initialize the processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Process the image and generate a summary (caption)
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    summary = processor.decode(out[0], skip_special_tokens=True)

    return summary
def analyze_folder_of_images(folder_path):
    summaries = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            summary = summarize_image(image)
            summaries.append({"filename": filename, "summary": summary})
    return summaries

def main():
    st.title("Automated Dataset Summerization, EDA & Data Modeling with Image Summarization")
    st.write("Upload a dataset or an image to get started.")

    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xls", "xlsx", "json", "xml"])

    image_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    image_folder = st.file_uploader("Upload a folder of images", type=None, accept_multiple_files=True)

    if uploaded_file is not None:
       
        dataset_name = uploaded_file.name
        df = load_data(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())

        # Menu for different operations
        operation = st.sidebar.selectbox(
            "Choose an operation",
            ["Summarization", "Normality Test", "Model Prediction", "Exploratory Data Analysis", "Outlier Detection and Removal"]
        )

        if operation == "Summarization":
            summary = generate_summary(df)
            ai_summary, labels = summarize_and_label(df, dataset_name)
            st.write("### Summary")
            st.text(summary)
            st.write("### AI-generated Summary")
            st.write(ai_summary)
            st.write("### Labels")
            st.write(labels)

        elif operation == "Normality Test":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            column = st.selectbox("Select a column to test for normality", numeric_columns)
            if st.button("Run Normality Tests"):
                col_data = df[column].dropna()
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                sns.histplot(col_data, kde=True, ax=ax[0])
                ax[0].set_title('Histogram')
                stats.probplot(col_data, dist="norm", plot=ax[1])
                ax[1].set_title('Q-Q Plot')
                st.pyplot(fig)
                shapiro_test = stats.shapiro(col_data)
                st.write("Shapiro-Wilk Test:")
                st.write("Statistic:", shapiro_test.statistic)
                st.write("p-value:", shapiro_test.pvalue)
                ks_test = stats.kstest(col_data, 'norm', args=(np.mean(col_data), np.std(col_data)))
                st.write("\nKolmogorov-Smirnov Test:")
                st.write("Statistic:", ks_test.statistic)
                st.write("p-value:", ks_test.pvalue)
                alpha = 0.05
                if shapiro_test.pvalue > alpha and ks_test.pvalue > alpha:
                    st.write("\nThe dataset follows a normal distribution (fail to reject H0).")
                else:
                    st.write("\nThe dataset does not follow a normal distribution (reject H0).")

        elif operation == "Model Prediction":
            target_column = st.selectbox('Select the target column', df.columns)
            if target_column:
                X = df.drop(columns=[target_column])
                y = df[target_column]
                if st.checkbox('Handle missing values'):
                    imputer_X = SimpleImputer(strategy='mean')
                    X = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns)
                    imputer_y = SimpleImputer(strategy='most_frequent' if y.dtype == 'object' else 'mean')
                    y = pd.Series(imputer_y.fit_transform(y.values.reshape(-1, 1)).flatten(), name=target_column)
                categorical_features = X.select_dtypes(include=['object']).columns
                if not categorical_features.empty:
                    for column in categorical_features:
                        le = LabelEncoder()
                        X[column] = le.fit_transform(X[column])
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                if st.checkbox('Scale features'):
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                if st.checkbox('Generate polynomial features'):
                    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
                    X = poly.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Automatically detect model type
                if pd.api.types.is_numeric_dtype(y):
                    model_type = 'Regressor'
                    tpot = TPOTRegressor(verbosity=2, generations=5, population_size=20, random_state=42)
                else:
                    model_type = 'Classifier'
                    tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42)

                if st.button('Run Model Predicator'):
                    tpot.fit(X_train, y_train)
                    y_pred = tpot.predict(X_test)
                    
                    if model_type == 'Classifier':
                        st.write("### Classification Report")
                        st.write(classification_report(y_test, y_pred))
                        st.write("### Accuracy Score")
                        st.write(accuracy_score(y_test, y_pred))
                    else:
                        st.write("### Mean Squared Error")
                        st.write(mean_squared_error(y_test, y_pred))
                        st.write("### R-squared Score")
                        st.write(r2_score(y_test, y_pred))
                    st.write("### Best Pipeline")
                    st.write(tpot.fitted_pipeline_)
                    st.text(tpot.fitted_pipeline_)
                    
        elif operation == "Exploratory Data Analysis":
            st.write("### Exploratory Data Analysis (EDA)")
            remove_outliers_option = st.checkbox('Remove Outliers')
            if st.button("Generate EDA Report"):
                with st.spinner("Generating EDA report..."):
                    show_results(df, remove_outliers_option)

        elif operation == "Outlier Detection and Removal":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            column = st.selectbox("Select a column to detect outliers", numeric_columns)
            if st.button("Detect Outliers"):
                outliers = detect_outliers(df[[column]])
                st.write(f"Outliers in column '{column}':")
                st.write(df[outliers])

                # Plot the column with outliers highlighted
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=df[column], ax=ax)
                ax.set_title(f'Box Plot of {column} with Outliers')
                st.pyplot(fig)
                
            if st.button("Remove Outliers"):
                df = remove_outliers(df)
                st.write("Outliers removed.")
                st.write("Updated data preview:")
                st.write(df.head())
        elif operation == "Chat with AI":
            st.write("### Chat with AI")
            chat_input = st.text_input("Ask a question about your dataset:")
            if st.button("Send"):
                if chat_input:
                    context = f"Dataset {dataset_name} contains columns: {', '.join(df.columns)}."
                    ai_response = generate_answer(context, chat_input)
                    st.write("### AI Response")
                    st.write(ai_response)
                    
                    # Maintain chat history
                    st.session_state['messages'].append({"role": "user", "content": chat_input})
                    st.session_state['messages'].append({"role": "assistant", "content": ai_response})

                for message in st.session_state['messages']:
                    if message['role'] == 'user':
                        st.write(f"**You:** {message['content']}")
                    else:
                        st.write(f"**AI:** {message['content']}")

    elif image_file is not None:
        try:
            image = Image.open(image_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            summary = summarize_image(image)
            st.write("### Image Summary")
            st.text(summary)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    elif image_folder is not None:
        try:
            folder_path = 'uploaded_images'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            for img_file in image_folder:
                with open(os.path.join(folder_path, img_file.name), "wb") as f:
                    f.write(img_file.getbuffer())
            st.write("### Uploaded Folder of Images")
            summaries = analyze_folder_of_images(folder_path)
            for summary in summaries:
                st.write(f"**{summary['filename']}**: {summary['summary']}")
        except Exception as e:
            st.error(f"Error processing folder of images: {str(e)}")
if __name__ == "__main__":
    main()
