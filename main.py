
"""
AI-based Document Transformation POC
Converts unstructured inputs to structured JSON, PDF, and summary
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Core libraries
import pytesseract
from PIL import Image
import cv2
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentTransformer:
    """
    Main class for transforming unstructured documents to structured formats
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the document transformer
        
        Args:
            openai_api_key: OpenAI API key for advanced text processing
        """
        self.openai_client = None
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Configure Tesseract path (adjust based on your system)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
        
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf', '.txt']
        
    
    

    
    
    def structure_content_with_ai(self, text: str) -> Dict[str, Any]:
        """
        Use AI to structure the extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Structured content as dictionary
        """
        if not self.openai_client:
            # Fallback to basic structuring
            return self.structure_content_basic(text)
        
        try:
            prompt = f"""
            Analyze the following text and structure it into a JSON format with the following schema:
            {{
                "title": "Main title or topic",
                "sections": [
                    {{
                        "header": "Section header",
                        "content": "Section content",
                        "type": "paragraph/list/table"
                    }}
                ],
                "key_points": ["List of key points"],
                "entities": {{
                    "people": ["Names mentioned"],
                    "dates": ["Dates mentioned"],
                    "locations": ["Locations mentioned"],
                    "organizations": ["Organizations mentioned"]
                }},
                "metadata": {{
                    "word_count": "number",
                    "estimated_reading_time": "X minutes",
                    "document_type": "meeting_notes/letter/report/other"
                }}
            }}
            
            Text to analyze:
            {text}
            
            Return only the JSON structure, no additional text.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a document analysis expert. Structure the given text into the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            structured_data = json.loads(response.choices[0].message.content)
            return structured_data
            
        except Exception as e:
            logger.error(f"Error structuring content with AI: {str(e)}")
            return self.structure_content_basic(text)
    
    def structure_content_basic(self, text: str) -> Dict[str, Any]:
        """
        Basic text structuring without AI
        
        Args:
            text: Raw extracted text
            
        Returns:
            Structured content as dictionary
        """
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        # Basic structure
        structured = {
            "title": lines[0] if lines else "Untitled Document",
            "sections": [],
            "key_points": [],
            "entities": {
                "people": [],
                "dates": [],
                "locations": [],
                "organizations": []
            },
            "metadata": {
                "word_count": len(text.split()),
                "estimated_reading_time": f"{max(1, len(text.split()) // 200)} minutes",
                "document_type": "other"
            }
        }
        
        # Group lines into sections
        current_section = {"header": "Content", "content": "", "type": "paragraph"}
        
        for line in lines[1:]:  # Skip title
            if len(line) > 0:
                if line.isupper() or line.endswith(':'):
                    # Potential header
                    if current_section["content"]:
                        structured["sections"].append(current_section)
                    current_section = {"header": line, "content": "", "type": "paragraph"}
                else:
                    current_section["content"] += line + " "
        
        if current_section["content"]:
            structured["sections"].append(current_section)
        
        # Extract key points (lines starting with bullet points or numbers)
        for line in lines:
            if line.startswith(('•', '-', '*')) or (len(line) > 2 and line[0].isdigit() and line[1] in '.)'): 
                structured["key_points"].append(line)
        
        return structured
    
    def generate_summary(self, text: str) -> str:
        """
        Generate a summary of the content
        
        Args:
            text: Full text content
            
        Returns:
            Summary text
        """
        if not self.openai_client:
            # Basic summary (first 150 words)
            words = text.split()
            if len(words) <= 150:
                return text
            return ' '.join(words[:150]) + "..."
        
        try:
            prompt = f"""
            Generate a concise summary of the following text in 100-150 words. 
            Focus on the main topics, key points, and important information:
            
            {text}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert summarizer. Create concise, informative summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            words = text.split()
            return ' '.join(words[:150]) + "..." if len(words) > 150 else text
    



# MAIN FILE STARTS HERE 

import cv2
import numpy as np

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # # Save the cleaned image
    # output_path = "cleaned_output.png"
    # cv2.imwrite(output_path, cleaned)
    # print(f"Cleaned image saved to {output_path}")

    # Display the cleaned image
    #cv2.imshow("Cleaned Image", cleaned)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return cleaned


def extract_text_from_image(image_path: str) -> str:

    # Preprocess image
    processed_img = preprocess_image(image_path)
    
    # Convert numpy array to PIL Image 
    pil_img = Image.fromarray(processed_img)
    
    # Perform OCR with custom config
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:()[]{}"-/\n '
    text = pytesseract.image_to_string(pil_img, config=custom_config)

    with open("text_extract/{}.txt".format(os.path.splitext(os.path.basename(image_path))[0]), "w") as file:
        file.write(text)
        
    
    return text.strip()


    


image_path = "images\prescription1.jpg"

print(extract_text_from_image(image_path))