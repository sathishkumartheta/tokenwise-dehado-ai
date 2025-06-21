#!/usr/bin/env python
# coding: utf-8

# # Inference.py
# - **Instantiation:** form = FormInference()
# - **Method:** form.inference([image_path], [output_path]) (output path is optional)

# In[2]:


import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle
import json
from PIL import ImageEnhance
import PIL.Image as Img
import re
from IPython import display
display.clear_output()

import ultralytics
import os
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image
import time
import psutil
import torch
from transformers import TrOCRProcessor, default_data_collator, VisionEncoderDecoderModel

from custom_functions import run_yolo_inference, extract_fields, extract_text, transform_json, load_ref_coordinates


# In[3]:


class FormInference:
    def __init__(self, device='cpu'):
        """
        Initialize the inference class by loading the model.
        """
        self.device = device        
        self.ref_table_coordinates = load_ref_coordinates("../data/Ref_Coordinates.xlsx")
        with open('../data/field_vocabulary.json', 'r') as f:
            self.field_vocab = json.load(f)
        print("Loading Yolo Model...")
        self.model = self.load_yolo_model("../model/Yolov8s_Custom.pt")
        print("Loading OCR Model...")
        self.processor, self.ocr_model = self.load_ocr_model()
        self.ocr_model.to(self.device)
        print("Initialization is completed")

    def load_yolo_model(self, model_path):
        """
        Load the trained model.
        """
        model = YOLO(model_path)
        return model

    def load_ocr_model(self):
        """
        Load the trained OCR model.
        """
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten", use_fast=True)
        ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        return processor, ocr_model

    def inference(self, image_path, out_path = "output.json"):
        print(f"Inferencing on image at location {image_path} is initiated..")
        print("Running detection Model..")
        pred_boxes = run_yolo_inference(self.model, image_path)
        print("Form field extraction in progress..")
        pred_field_boxes = extract_fields(image_path, self.ref_table_coordinates, pred_boxes)
        print("Running OCR..")
        pred_field_text = extract_text(image_path, pred_field_boxes, self.processor, self.ocr_model, self.field_vocab)
        json_content = transform_json(pred_field_boxes, pred_field_text)
        print("Writing to JSON..")
        with open(out_path, 'w') as json_file:
            json.dump(json_content, json_file, indent=4)
        print("Form Extraction Completed..")


# In[ ]:




