import openai
from CollectDataForTarget.CollectData import read_text
from dotenv import load_dotenv
import os


"""
This is a draft as well (mostly from GPT itself); I haven't made the payment, 
so we can't use it yet. Just an idea of how to use LM to get anwers. 
(openai.error.RateLimitError: You exceeded your current quota, please check your plan and billing details.)

This file evaluates a food based on ingredient list, while the other things 
we've been doing deals with nutrition label. 

Theoretically, if the rating in nutrition or ingredient is below a certain 
threshold, we label it as bad. If we have time, we can do allergens too 
using keywords from LM.
"""

class LanguageModel:
    def author():
        """
        Returns:
        author name in list
        """
        return ['Rachel Yu-Wei Lai', 'Simon Cheng-Wei Huang']

###########################################################
    def __init__(self, api_key):
        self.OPEN_AI_ACCESS_KEY = api_key

    def language_model_ingredients(self, file_name: str) -> str:
        """
        This function takes in a file name of ingredient list, reads the text, then enters the
        info for the language model to evaluate, which will end up producing a score.
        ------------------
        Parameters: 
        file_name: file name of ingredient list in str
        ------------------
        Returns: chatGPT's response in str
        """
        
        # Set your API key
        api_key = self.OPEN_AI_ACCESS_KEY

        # Define your prompt (the text you want to provide as input)
        ingredients = read_text(file_name)
        print(ingredients)
        prompt = f"Rate the healthiness of the food on a scale of 1 to 10 based on its nutritional content provided (1 is most healthy and 10 is most unhealthy), give me only the numeric answer:  {ingredients}."

        # Make an API request
        response = openai.ChatCompletion.create(
            api_key=api_key,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI trained to evaluate the healthiness of food based on its nutritional information. Consider factors such as total fat, saturated fat, cholesterol, sodium, carbohydrates, dietary fiber, sugars, protein, and vitamins and minerals. Provide a healthiness score out of 10, taking into account that a balanced food item should have low saturated fat, low sodium, high fiber, and contain essential vitamins and minerals. you will only give numeric answer about the health level, and 1 is most healthy"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-3.5-turbo",
            max_tokens=10,
        )

        # Get the generated response
        answer = response.choices[0].message.content

        # print("Answer:", answer)
        
        return answer

##########################################################
if __name__ == "__main__":
    model = LanguageModel()
    file_name_ingredients = "INPUTS/foodlabel.png"
    model.language_model_ingredients(file_name_ingredients)