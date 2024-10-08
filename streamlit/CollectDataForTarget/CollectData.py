import tabula as tb
import pandas as pd
import os
from PIL import Image
import easyocr
import uuid


def author():
    """
    Returns:
       author name in list
    """
    return ['Rachel Yu-Wei Lai']

###########################################################
def read_text(file:str) -> list:
    """
    This function uses easyocr, and OCR framework to read text from a file, given the 
    file name.
    ------------------
    Parameters: 
    file_name: file name in str, currently called foodlabel.png, subject to future changes
    ------------------
    Returns: text in list
    """
    reader = easyocr.Reader(['en'])
    text = reader.readtext(file, detail = 0, text_threshold=0.7)
    return text
        
def convert_image_to_pdf() -> None:
    """
    This function converts all png, jpg, or jpeg images files to PDF for better processing 
    in the future.
    ------------------
    Parameters: None
    ------------------
    Returns: None
    """
    output_dir = 'OUTPUTS'
    source_dir = 'INPUTS'
    for file in os.listdir(source_dir):
        if file.split('.')[-1] in ('png', 'jpg', 'jpeg'):
            image = Image.open(os.path.join(source_dir, file))
            image_converted = image.convert('RGB')
            image_converted.save(os.path.join(output_dir, '{0}.pdf'.format(file.split('.')[-2])))
    return

def extract_info_from_text(text: list, category: str) -> dict:
    """
    Using the text output from read_text function, extract info and convert it into a 
    pandas df. Each time we run the read_text function, the output will become one row in 
    the final df. The idea is to get ~200 entries/rows (aka food items) in our database. 
    Maybe store the data in AWS or snowflake? Let me know what you guys think.
    ------------------
    Parameters: 
    text: text in str
    category: category of food in str
    ------------------
    Returns: single-row table with columns as features in pd.DataFrame
    """
    """ 
    features: 
    Calories from added sugar/total calories
    Calories from fat/total calories
    Calories from protein/total calories
    Calories from carbs/total calories
    Calories from saturated fat/total calories
    Calories from trans fat/total calories
    more to be added
    """
    nutrition_map = {}
    nutrition_map['Category'] = category
    for i in range(len(text)):
        if text[i] == 'Calories':
            nutrition_map['Calories'] = int(text[i+1])
            continue
        if 'Total Fat' in text[i]:
            nutrition_map['Total Fat'] = int(text[i].split(' ')[-1][:(len(text[i]) - 1)][:-1])
            continue
        if 'Saturated Fat' in text[i]:
            nutrition_map['Saturated Fat'] = float(text[i].split(' ')[-1][:(len(text[i]) - 1)][:-1])
            continue
        if 'Trans Fat' in text[i]:
            tmp = text[i].split(' ')[-1][:(len(text[i]) - 1)][:-1]
            tmp = 0 if tmp == 'O' else int(tmp)
            nutrition_map['Trans Fat'] = tmp
            continue
        if 'Total Carbohydrate' in text[i]:
            nutrition_map['Total Carbohydrate'] = int(text[i].split(' ')[-1][:(len(text[i]) - 1)][:-1])
            continue
        if 'Added Sugar' in text[i]:
            nutrition_map['Added Sugar'] = int(text[i].split(' ')[-3][:(len(text[i]) - 3)][:-1])
            continue
        if 'Protein' in text[i]:
            nutrition_map['Protein'] = int(text[i+1][:-1])
            continue
    return nutrition_map


def convert_info_to_df(d: dict)-> pd.DataFrame:
    nutrition_map = {}
    nutrition_map['category'] = d['Category']
    # Calories from Added Sugar vs Total Calories
    nutrition_map['suga_to_total'] = round(d['Added Sugar'] * 4 / d['Calories'], 4)
    # Calories from Fat vs Total Calories
    nutrition_map['fat_to_total'] = round(d['Total Fat'] * 9 / d['Calories'], 4)
    # Calories from Protein vs Total Calories
    nutrition_map['pro_to_total'] = round(d['Protein'] * 4 / d['Calories'], 4)
    # Calories from Carbohydrates vs Total Calories
    nutrition_map['carb_to_total'] = round(d['Total Carbohydrate'] * 4 / d['Calories'], 4)
    # Calories from Saturated Fat vs Total Calories
    nutrition_map['satu_to_total'] = round(d['Saturated Fat'] * 9 / d['Calories'], 4)
    # Calories from Trans Fat vs Total Calories
    nutrition_map['tran_to_total'] = round(d['Trans Fat'] * 9 / d['Calories'], 4)
    res = pd.DataFrame(nutrition_map, index=[str(uuid.uuid4())])
    print(res)
    return res

def add_category_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add name of food and score from the website: 
    https://www.ewg.org/foodscores/products/888849006045-QuestProteinCookiePeanutButter/
    ------------------
    Parameters: single-row info table in pd.DataFrame
    ------------------
    Returns: single-row table with columns as features, name, and score in pd.DataFrame
    """
    return

###########################################################
if __name__ == "__main__":
    # convert_image_to_pdf()
    text = read_text('foodlabel.png')
    print(extract_info_from_text(text, 'food1'))
    # convert_info_to_df(extract_info_from_text(text, 'food1')).to_csv('OUTPUTS/test.csv')