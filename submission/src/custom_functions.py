#!/usr/bin/env python
# coding: utf-8

# # custom_functions.py: Utility Function

# In[ ]:


import re
import cv2
import pandas as pd
import numpy as np
from PIL import ImageEnhance
import PIL.Image as Img


# # Functions

# ### Generic

# In[2]:


def compute_iou(boxA, boxB):
    # Unpack coordinates
    xA_min, yA_min, xA_max, yA_max = boxA
    xB_min, yB_min, xB_max, yB_max = boxB

    # Compute intersection coordinates
    x_left = max(xA_min, xB_min)
    y_top = max(yA_min, yB_min)
    x_right = min(xA_max, xB_max)
    y_bottom = min(yA_max, yB_max)

    # Check for no overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Compute intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute areas of each box
    boxA_area = (xA_max - xA_min) * (yA_max - yA_min)
    boxB_area = (xB_max - xB_min) * (yB_max - yB_min)

    # Compute IoU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou


# In[32]:


def transform_key(field):
    if "Father" in field:
        return field.replace("Fatherhusbandname", "Father/husbandname")
    else:
        return field

def transform_json(pred_field_boxes, pred_field_text):
    content = []
    for key, val in pred_field_boxes.items():
        val = [round(ele) for ele in val]
        content.append({"Field name": transform_key(key), "Field value": pred_field_text[key], "Coordinate": val})
    return content


# ### Yolo

# In[4]:


def run_yolo_inference(model, image_file):
    results = model(image_file, conf=0.3, iou=0.4, save=True)
    if not results[0].boxes is None:
        boxes = results[0].boxes.xyxy.tolist()
    else:
        boxes = []
        print("No handwritten text detected")
    return boxes


# ### Field Matching

# In[5]:


def get_table_coordinates(image_file):
    image = Img.open(image_file)
    enhancer = ImageEnhance.Contrast(image)
    contrast_image = enhancer.enhance(2)  
    opencv_image = np.array(contrast_image) # Enhance Contrast 
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR) # Convert RGB to BGR    
    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY) # Convert BGR to GRAY    
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0) # Apply Blurr on Image    
    edged = cv2.Canny(blurred_image, 150, 200) # Apply Canny Edge Detection
    
    kernel = np.ones((3,3), np.uint8)
    dilated_img = cv2.dilate(edged, kernel, iterations=1) # Perform dilation    
    contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours

    area = []
    for cont in contours:
        bbox_coord = cv2.boundingRect(cont)
        area.append(bbox_coord[2] * bbox_coord[3])

    # Get Coordinates of contour with Max Contour Area    
    cnt = contours[np.argmax(area)]
    x, y, w, h = cv2.boundingRect(cnt)
    x1, x2, y1, y2 = x, x + w, y, y + h
    return [x1, y1, x2, y2] #, contours, hierarchy


# In[6]:


def norm_coordinates(table_coord, field_coord):
    cell_x1, cell_y1, cell_x2, cell_y2 = field_coord
    table_x1, table_y1, table_x2, table_y2 = table_coord
    
    table_width = table_x2 - table_x1
    table_height = table_y2 - table_y1
    
    norm_x1 = (cell_x1 - table_x1) / table_width
    norm_y1 = (cell_y1 - table_y1) / table_height
    norm_x2 = (cell_x2 - table_x1) / table_width
    norm_y2 = (cell_y2 - table_y1) / table_height
    
    return [np.round(norm_x1, 4), np.round(norm_y1, 4), np.round(norm_x2, 4), np.round(norm_y2, 4)]


# In[7]:


def load_ref_coordinates(file_path):
    ref_df = pd.read_excel(file_path)
    ref_df.set_index('Field', inplace=True)
    ref_table_coordinates = list(ref_df.loc['Full'])[:-2]
    ref_field_coordinates = {}
    ref_columns = list(ref_df.index)
    for ind in range(1, len(ref_columns)):
        f_coordinates = list(ref_df.loc[ref_columns[ind]])[:-2]        
        ref_field_coordinates[ref_columns[ind]] = norm_coordinates(ref_table_coordinates, f_coordinates)
    return ref_field_coordinates


# In[8]:


def find_matches(table_coordinates, field_coordinates):
    pred_field = []
    for cent in field_coordinates:
        min_dist = {}
        for key, val in table_coordinates.items():
            min_dist[key] = compute_iou(cent, val)
        pred_field.append(max(min_dist, key = lambda x: min_dist[x]))
    
    return pred_field


# In[9]:


def extract_fields(image_file, ref_table_coordinates, pred_boxes):
    local_table_coordinates = get_table_coordinates(image_file)    
    field_coordinates = []
    for coord in pred_boxes:
        field_coordinates.append(norm_coordinates(local_table_coordinates, coord))
    pred_fields = find_matches(ref_table_coordinates, field_coordinates)
    pred_results = {f:c for f, c in zip(pred_fields, pred_boxes)}
    return pred_results


# ### TrOCR

# In[10]:


def run_ocr(image, processor, ocr_model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values) # Generate output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # Decode generated token ids to string
    return generated_text


# In[11]:


def extract_text(image_file, field_boxes, processor, ocr_model, field_vocab):
    img = cv2.imread(image_file)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    field_text = {}
    for field, box in field_boxes.items():
        box = [int(coord) for coord in box]
        start_x, start_y, end_x, end_y = box
        cropped_image = image_rgb[start_y:end_y, start_x:end_x]
        text = run_ocr(cropped_image, processor, ocr_model)
        field_text[field] = post_processing([field, text], field_vocab)
    return field_text


# ### Post Processing

# In[12]:


def remove_special_char(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9]+', '', text)
    return cleaned_text
def numeric_only(text):
    cleaned_text = re.sub(r'[^0-9]+', '', text)
    return cleaned_text
def char_space(text):
    cleaned_text = re.sub(r'[^A-Za-z ]+', '', text)
    return cleaned_text.strip()
def generic(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9 +\-]+', '', text)
    return cleaned_text.strip()
def alphabets_only(text):
    cleaned_text = re.sub(r'[^A-Za-z]+', '', text)
    return cleaned_text.capitalize()
def alphanum_specialchar(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9 +\-/,.]+', '', text)
    return cleaned_text.strip()


# In[13]:


def lang_specific(text, field_vocab, camelcase=True):
    text = re.sub(r'[^A-Za-z ,]+', '', text)
    text = re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', text) # Remove special character at start and end
    text = re.sub(r'[^A-Za-z0-9, ]+', ',', text) # Remove single special character to comma
    text = re.sub(r'\s+,', ',', text) # Remove extra spaces before comma
    words = []
    for word in text.split():
        if camelcase:
            capital_letters = sum(1 for char in text if char.isupper())
            if capital_letters <= len(word)//2:
                word = word.capitalize().strip()
            else:
                word = word.strip()
        else:
            word = word.strip()
        word = autocorrect_words(word.replace(",",""), 'language', field_vocab) + ","
        words.append(word)
    return " ".join(words)[:-1]
    
def date_format(text):
    elements = text.split("/")
    corrected = []
    if len(elements) == 3:
        corrected = [str(int(ele)) for ele in elements]
        return "/".join(corrected)
    else:
        return text
        
def date_specific(text):
    cleaned_text = re.sub(r'[^0-9]+', '/', text)
    cleaned_text = re.sub(r'/+', '/', cleaned_text)
    cleaned_text = re.sub(r'^[^0-9]+|[^0-9]+$', '', cleaned_text)
    cleaned_text = date_format(cleaned_text)
    return cleaned_text

def bloodgroup_specific(text):
    cleaned_text = re.sub(r'[^ABOTabot0-1+\-]+', '', text)
    cleaned_text = cleaned_text.replace('t', '+').replace('0', 'O')
    return cleaned_text.upper().strip()

def address_specific(text):
    text = re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', text) # Remove special character at start and end
    text = re.sub(r'[^A-Za-z0-9, /\-.]+', ',', text) # Remove single special character to comma
    text = re.sub(r'\s+,', ',', text) # Remove extra spaces before comma
    return text

def reference_specific(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9 \-]+', '', text)
    words = cleaned_text.split("-")
    if len(words) == 1:        
        cleaned_text = " - ".join([re.sub(r'[^A-Za-z]+', '', text), re.sub(r'[^0-9]+', '', text)])
    elif len(words) == 2:
        cleaned_text = " - ".join([process_words(words[0], True).strip(), re.sub(r'[^0-9]+', '', words[1])])    
    return cleaned_text
        
def pan_specific(text):
    text = remove_special_char(text)
    text_transform = ""
    if len(text) == 10:
        text_transform = num2alpha(text[:5])
        text_transform += alpha2num(text[5:-1])
        text_transform += num2alpha(text[-1])
    else:
        text_transform = text    
    return text_transform.upper()


# In[14]:


def autocorrect_words(text, field, field_vocab):
    accept_flag = False
    if field in field_vocab:
        training_labels = field_vocab[field]
        dist = [levenshtein_distance(text, ele) for ele in training_labels]
        
        min_dist, min_dist_ind = np.min(dist), np.argmin(dist)
        corrected = training_labels[min_dist_ind]
        if ((min_dist <= 1) or (min_dist <= 2 and len(corrected) > 4) or (min_dist <= 3 and len(corrected) > 8)) and dist.count(min_dist) <= 1:
            return corrected
        else:
            return text
    else:
        return text
def process_words2(text, field_vocab):
    words = []
    for word in text.split():
        word = word.capitalize().strip()
        word = autocorrect_words(word, field, field_vocab)
        words.append(word)
    return " ".join(words)

def process_words(text, camelcase=False):
    words = []
    for word in text.split():
        if camelcase:
            capital_letters = sum(1 for char in text if char.isupper())
            if capital_letters <= len(word)//2:
                word = word.capitalize()
            else:
                word = word.strip()
        else:
            word = word.strip()
        words.append(word)
    return " ".join(words)


# In[15]:


alphatonum = {"o": "0", "O": "0", "b": "6", "z": "2", "Z": "2", "B": "8", "G": "6", "I": "1", "T":"7"}
numtoalpha = {val: key for key, val in alphatonum.items()}
schartonum = {"/": "1", "\\": "1", "(": "1"}


def alpha2num(text):
    text_transform = ""
    for char in text:
        if char.isalpha():
            if char in alphatonum:
                text_transform += alphatonum[char]
            else:
                text_transform += char
        else:
            text_transform += char
    return text_transform

def num2alpha(text):
    text_transform = ""
    for char in text:
        if char.isdigit():
            if char in numtoalpha:
                text_transform += numtoalpha[char]
            else:
                text_transform += char
        else:
            text_transform += char
    return text_transform

def special2num(text):
    text_transform = ""
    for char in text:
        if not char.isalnum():
            if char in schartonum:
                text_transform += schartonum[char]
            else:
                text_transform += char
        else:
            text_transform += char
    return text_transform


# In[16]:


def post_processing(data, field_vocab):
    if data[0] in ['Dateofbirth', 'date']:
        return date_specific(data[1])
    elif data[0] in ['nationality', 'gender', 'maritalstatus']:
        clean_text = alphabets_only(data[1])
        return autocorrect_words(clean_text, data[0], field_vocab)    
    elif data[0] in ['candidatename', 'Fatherhusbandname']:
        clean_text = char_space(data[1])
        return process_words(clean_text, True)
    elif data[0] in ['place']:
        clean_text = alphabets_only(data[1])
#         clean_text = process_words(clean_text, True)
        return autocorrect_words(clean_text, data[0], field_vocab)
    elif data[0] in ['contactnumber', 'AlternateNo', 'aadhaarcard']:
        clean_text = special2num(alpha2num(data[1]))
        return numeric_only(clean_text)
    elif data[0] in ['permanentaddress', 'presentaddress']:
        clean_text = alphanum_specialchar(data[1])
        clean_text = address_specific(clean_text)
        return process_words(clean_text, True)
    elif data[0] in ['bloodgroup']:
        return bloodgroup_specific(data[1])
    elif data[0] in ['experience', 'experience1']:
        clean_text = generic(data[1])
        return process_words(clean_text, False)
    elif data[0] in ['qualification']:
        clean_text = generic(data[1])
        clean_text = process_words(clean_text, True)        
        return autocorrect_words(clean_text, data[0], field_vocab) 
    elif data[0] in ['referencescmob1', 'referencescmob2']:
        clean_text = reference_specific(data[1])
        return clean_text
    elif data[0] in ['pancard']:
        return pan_specific(data[1])
    elif data[0] in ['languageknown']:
        return lang_specific(data[1], field_vocab)
    else:
        return ""


# ### Evaluation Functions

# In[17]:


# --- METRICS & EFFICIENCY ---
def levenshtein_distance(s1, s2):
    if isinstance(s1, str):
        s1 = list(s1)
    if isinstance(s2, str):
        s2 = list(s2)
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def word_error_rate(y_true, y_pred):
    total_words = 0
    total_errors = 0
    for t, p in zip(y_true, y_pred):
        t_words = t.split()
        p_words = p.split()
        total_words += len(t_words)
        total_errors += levenshtein_distance(t_words, p_words)
    return total_errors / total_words if total_words > 0 else 0.0

def char_error_rate(y_true, y_pred):
    total_chars = 0
    total_errors = 0
    for t, p in zip(y_true, y_pred):
        total_chars += len(t)
        total_errors += levenshtein_distance(t, p)
    return total_errors / total_chars if total_chars > 0 else 0.0

def field_accuracy(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)

def document_level_accuracy(y_true, y_pred, doc_ids):
    from collections import defaultdict
    doc_true = defaultdict(list)
    doc_pred = defaultdict(list)
    for doc_id, t, p in zip(doc_ids, y_true, y_pred):
        doc_true[doc_id].append(t)
        doc_pred[doc_id].append(p)
    correct_docs = 0
    for doc_id in doc_true:
        if doc_true[doc_id] == doc_pred[doc_id]:
            correct_docs += 1
    return correct_docs / len(doc_true) if doc_true else 0.0

