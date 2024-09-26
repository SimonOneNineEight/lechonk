import streamlit as st
from PIL import Image
from CollectDataForTarget.CollectData import *
from Model.LinRegModel import *
from Model.DTRegressor import *
from Model.LangModel import *
import re

def extract_integer_from_string(input_string):
    match = re.search(r'\b\d+\b', input_string)
    if match:
        return int(match.group())
    else:
        return None

def main():
    
    # Create a title for your app
    st.title("Food Nutrition Analyzer")
    st.write("Welcome to our streamlit dashboard that takes food nutrition analysis and recommendation to the next level")    
    st.title('Upload Photo')
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Photo', use_column_width=False)
    
    if not uploaded_file:
        return
    
    api_key = st.text_area("Please enter your openAI API key for the language model")
    st.markdown("**Privacy and Security Notice**: We respect your privacy and are committed to protecting it. The OpenAI API key you provide will only be used to access OpenAI's services as required by this application. Your API key will not be stored, shared, or used for any purposes other than those explicitly stated here. We implement standard security measures to ensure the confidentiality and integrity of your API key during its use. You retain full control over your API key and can revoke our access at any time by resetting it within your OpenAI account settings. By providing your API key, you consent to its use as described above.")
    # convert PIL image into np.array for model use
    image_np = np.array(image)
    if st.button('Analyze Food Label'):
        try:
            text_result = read_text(image_np)
            
            st.subheader('Extracted Text:')
            for text in text_result:
                st.write(text)  # Display the recognized text

        except Exception as e:
            st.error(f"Error: {e}")
            st.write("Please upload another image")
            return
    
    if not text_result:
        return 
    
    st.subheader('Extracted Info:')
    print(text_result)
    extracted_info = extract_info_from_text(text_result, 'meals')
    st.write(extracted_info)
    
    st.subheader('Organized Data for Analysis:')
    converted_df = convert_info_to_df(extracted_info)
    st.write(converted_df)
    
#############################

    st.title('Two Machine Learning Models')
    st.subheader('* We are choosing the model with ')

    # Using columns to display text side by side
    col1, col2 = st.columns(2)  # Split the layout into 2 columns
    
    # load Data for training
    data_for_model_training_raw = pd.read_csv('../INPUTS/data_for_model_training.csv')
    data_for_model_training = convert_info_to_df_databse_lr(data_for_model_training_raw)
    data_for_model_training = data_for_model_training.dropna()

    with col1:
        st.header('Linear Model')
        linear_model = LinearRegressionModel(data_for_model_training)
        linear_model.split_data(test_size=0.2, random_state=42)
        linear_model.train_model()
        mse, r2 = linear_model.evaluate_model()
        coefficients = linear_model.get_coefficients()

        st.write("**Mean Squared Error**", f": {mse}")
        st.write("**R-squared (R2) Score**", f": {r2}")
        st.write("**Model Coefficients:**")
        for feature, coef in coefficients.items():
            st.write(f"{feature}: {coef}")
        
    with col2:
        dt_model = DecisionTreeRegressorModel(data_for_model_training)
        dt_model.split_data(test_size=0.2, random_state=42)
        dt_model.train_model()
        mse, r2 = dt_model.evaluate_model()
        feature_importance = dt_model.get_feature_importance()

        st.header('Decision Tree Regressor')
        st.write("**Mean Squared Error**", f": {mse}")
        st.write("**R-squared (R2) Score**", f": {r2}")
        st.write("**Feature Importance:**")
        for feature, importance in zip(data_for_model_training.columns[1:], feature_importance):
            st.write(f"{feature}: {importance}")

    predicted_score = linear_model.model.predict(converted_df.drop('category', axis=1))[0]
    st.header('Estimated Score')
    st.subheader('From the Linear Regression Model:')
    st.write(f"*{predicted_score}*")

    ########################################################## 
    
    if not api_key:
        lang_score = predicted_score
    else:
        lang_model = LanguageModel(api_key)
        score = lang_model.language_model_ingredients(image_np)
        lang_score = extract_integer_from_string(score)
        st.subheader('From the Language Model:')
        st.write(f"*{lang_score}*")
    
    ##########################################################
    predicted_score = (int(predicted_score) + int(lang_score)) / 2
    
    if api_key:
        st.subheader('Taking the Average of Scores from Both Models:')
        st.write(f"*{predicted_score}*")
    else:
        st.subheader("The score is:")
        st.write(f"*{predicted_score}*")
        st.write("If you want a more precise score integrate the language model, please refresh the page and enter you api key")
   
    
    if predicted_score <= 3:
        st.subheader('This food is great. Nice choice!')
        return
    elif predicted_score <= 6 and predicted_score > 3:
        # st.write('This food is fine. You can consume it as is. However, here are a few options:')
        st.subheader("This food is fine. You can consume it as is.")
    else:
        # st.write('This food is not good for your health. Please take a look at healthier options:')
        st.subheader("This food is not good for your health")

    # category_value = converted_df['category'][0]
    # recommendation_dataset = data_for_model_training[data_for_model_training['category'] == category_value]
    # print("Duplicated -->", recommendation_dataset.index.duplicated())

    # # Ensuring the DataFrame is free of duplicated indices
    # recommendation_dataset = recommendation_dataset[~recommendation_dataset.index.duplicated(keep='first')]

    # # Resetting index after removing duplicates
    # recommendation_dataset = recommendation_dataset.reset_index(drop=True)

    # # Sorting by 'Score'
    # recommendation_dataset = recommendation_dataset.sort_values('Score', ascending=True)

    # # Assuming you want the top two recommendations with Scores less than 6
    # recommendations = recommendation_dataset[recommendation_dataset['Score'] < 6].head(2)
            
if __name__ == '__main__':
    main()
