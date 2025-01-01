
import time
import requests
import json
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

base_url = 'https://workflow.wholesalecarparts.com.au/api'

token = None
def login():
    global token
    payload = {
        "ApplicationName": "external-facebook-bot",
        "ApplicationSecret": "85sFSyZRew4WB9qYvvn8SbkWQuvX2xhxGuGm95Ece2UqbVjt9HYNaHqAt6yAHcFTpteyjWgwharKdN=="
    }
    url = base_url + '/auth/authenticate'

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        token = response.json().get('token')
        return token
    else:
        raise Exception('Failed to login to Workflow')

def ensure_logged_in():
    global token
    if not token:
        login()

def get_header():
    ensure_logged_in()
    return {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def make_request_with_retry(method, url, headers=None, data=None, max_retries=5, delay=5):
    """
    Makes a request and retries if rate-limiting (429) is encountered.
    """
    time.sleep(delay)
    retries = 0
    while retries < max_retries:
        response = requests.request(method, url, headers=headers, json=data)
        
        if response.status_code == 429:  # Rate limit exceeded
            wait_time = delay * (2 ** retries)  # Exponential backoff
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
        elif response.status_code == 401:  # Unauthorized, likely token expired
            print("Token expired or invalid. Re-authenticating...")
            login()
            headers = get_header()
        else:
            return response

    raise Exception(f"Max retries exceeded for {method} {url}")

def get_img_ids_by_ic_part_code(ic, part_code):
    url = base_url + '/images/search'
    payload = {
        "InterchangeCode": ic,
        "PartTypeCode": part_code,
        "IncludeItemsCheckedForDamagesAndIncorrectLabelling": True
    }

    response = make_request_with_retry("POST", url, headers=get_header(), data=payload)
    if response.status_code == 200:
        content = json.loads(response.content.decode('utf-8'))
        if len(content) == 0:
            # print("No image ids")
            return []
        else:
            return content[0]['imageIDs']
    else:
        print(f"Request failed: {response.status_code} - {response.text}")
        return []

def save_images(ic_num, part_code):
    image_ids = get_img_ids_by_ic_part_code(ic_num, part_code)
    try:
        print(f"Processing {len(image_ids)} images...")
        for idx, image_id in enumerate(image_ids):
            # print(image_id)
            img_base64_str = get_img_by_id(image_id)
            
            if img_base64_str is None:
                continue
            
            if ',' in img_base64_str:
                img_base64_str = img_base64_str.split(',')[1]
        
            # Decode the Base64 string
            image_bytes = base64.b64decode(img_base64_str)
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.size[0] < 1000:
                # print("Image resolution is too low to be saved")
                continue
            
            # Create directory if it doesn't exist
            directory = f"data_api/{part_code}"
            os.makedirs(directory, exist_ok=True)
            
            
            save_path = os.path.join(directory, f"{ic_num}_{idx}.png")
            
            # Check if the file already exists
            if os.path.exists(save_path):
                # print(f"File already exists: {save_path}. Skipping...")
                continue
            
            image.save(save_path)
            print(f"Image saved to: {save_path}")
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
               


def get_img_by_id(image_id):
    url = base_url + '/images/' + str(image_id)
    response = make_request_with_retry("GET", url, headers=get_header())
    if response.status_code == 200:
        try:
            return json.loads(response.content.decode('utf-8'))['data']
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None
    else:
        print(f"Request failed: {response.status_code} - {response.text}")
        return None

def preview_base64_image(base64_string):

    # Remove any prefix like "data:image/png;base64," if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    try:
        # Decode the Base64 string
        image_bytes = base64.b64decode(base64_string)
        
        # Open the image using Pillow
        image = Image.open(io.BytesIO(image_bytes))
        
        # Print out the resolution of the image
        # print(f"Resolution: {image.size[0]}x{image.size[1]}")
        
        if image.size[0] < 1000:
            print("Image res is too low to preview")
            return
        
        # Create a matplotlib figure
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.title('Base64 Image Preview')
        plt.show()
    
    except Exception as e:
        print(f"Error processing image: {e}")


df = pd.read_csv('updated_part_code_ic.csv')

for index, row in tqdm(df.iterrows(), total=len(df)):
    ic_num = str(row['interchangeNumber'])
    part_code = str(row['partTypeCode'])
    print("**********")
    if ic_num == 'nan' or part_code == 'nan':
        print(f"Skipping {ic_num}/{part_code}..")
        df.drop(index=index, inplace=True)
        df.to_csv('updated_part_code_ic.csv', index=False)
        continue
    print(f"Saving {part_code}/{ic_num}..")
    
    # Call the save_images function
    is_successful = save_images(ic_num, part_code)
    
    # If successful, drop the row and save the DataFrame immediately
    if is_successful:
        df.drop(index=index, inplace=True)
        df.to_csv('updated_part_code_ic.csv', index=False)