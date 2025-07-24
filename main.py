import json
import os
import pytesseract
from PIL import Image
import cv2
import numpy as np
import requests
import demjson3


def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    
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

def choose_better_text(image_path, text1, text2):

    if len(text1) > len(text2):
        text_to_write = text1
        return_text = 1
    else:
        text_to_write = text2
        return_text = 2
    with open("text_extract/{}.txt".format(os.path.splitext(os.path.basename(image_path))[0]), "w") as file:
        file.write(text_to_write)
    
    return text1 if return_text == 1 else text2


def extract_text_from_image(image_path: str) -> str:

    # Preprocess image
    processed_img = preprocess_image(image_path)

    # raw image
    raw_image = cv2.imread(image_path)
    
    # Convert numpy array to PIL Image 
    pil_img = Image.fromarray(processed_img)
    
    # Perform OCR with custom config
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:()[]{}"-/\n '
    text1 = pytesseract.image_to_string(pil_img, config=custom_config)
    text2 = pytesseract.image_to_string(raw_image, config=custom_config)

    text = choose_better_text(image_path, text1, text2)
    

    return text.strip()


def classify_document_type(text):
    """
    Classifies the document as 'prescription' or 'record' based on keywords.
    """
    prescription_keywords = [
        'prescription', 'rx', 'take', 'tablet', 'capsule', 'mg', 'dose', 'sig', 'refill', 'pharmacy', 'medication', 'dispense', 'prn', 'bid', 'tid', 'qid', 'po', 'od', 'bd', 'hs', 'before meals', 'after meals', 'doctor', 'dr.', 'physician', 'medicines', 'meds'
    ]
    record_keywords = [
        'record', 'patient', 'history', 'diagnosis', 'treatment', 'report', 'examination', 'findings', 'admission', 'discharge', 'summary', 'progress', 'notes', 'vitals', 'bp', 'pulse', 'temperature', 'complaint', 'investigation', 'lab', 'test', 'result', 'clinical', 'case', 'follow up', 'consultation'
    ]
    text_lower = text.lower()
    prescription_score = sum(1 for kw in prescription_keywords if kw in text_lower)
    record_score = sum(1 for kw in record_keywords if kw in text_lower)
    if prescription_score > record_score:
        return 'prescription'
    elif record_score > prescription_score:
        return 'record'
    else:
        return 'unknown'

def load_json_preset(doc_type):
    preset_path = os.path.join('json_presets', f'{doc_type}.json')
    if not os.path.exists(preset_path):
        return None
    with open(preset_path, 'r') as f:
        return json.load(f)

def ollama_map_text_to_json(text, doc_type):
    prompt = f"""
You are an expert at extracting structured data from unstructured text.
Given the following document type: {doc_type}.
Extract all relevant fields and their values from the text below and return them as a JSON object. Use field names that make sense for this document type. If a field is missing, omit it.

Text:
{text}

Return only the JSON object.
"""
    print("=== PROMPT SENT TO MODEL ===")
    print(prompt)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",  # or "phi", "llama2", etc.
            "prompt": prompt,
            "stream": False
        }
    )
    result = response.json()
    print("=== RAW MODEL RESPONSE ===")
    print(result['response'])
    import re
    match = re.search(r'({[\s\S]*})', result['response'])
    if match:
        return safe_json_loads(match.group(1))
    else:
        return None

def map_text_to_json(text, doc_type):
    # Use AI-based extraction only, let the model decide the fields
    ai_result = ollama_map_text_to_json(text, doc_type)
    if ai_result:
        return ai_result
    return None

def summarize_data(structured_json):
    import requests
    prompt = f"""
Summarize the following structured data in 100-150 words, highlighting the key content, topics covered, or instructions mentioned. Write in clear, professional language.

Data:
{json.dumps(structured_json, indent=2)}
"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",  # or "phi", "llama2", etc.
            "prompt": prompt,
            "stream": False
        }
    )
    result = response.json()
    import re
    match = re.search(r'(?s)\n*(.*)', result.get('response', ''))
    if match:
        return match.group(1).strip()
    return None


def create_pdf_from_json(structured_json, summary, output_path):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Document Type: {structured_json.get('document_type', 'Unknown')}", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Summary:\n{summary}")
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Extracted Data:", ln=True)
    pdf.set_font("Arial", '', 12)
    def render_json(data, indent=0):
        if isinstance(data, dict):
            for k, v in data.items():
                pdf.cell(indent * 5)
                pdf.multi_cell(0, 8, f"{k}:" if not isinstance(v, (dict, list)) else f"{k}:")
                render_json(v, indent + 1)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                pdf.cell(indent * 5)
                pdf.multi_cell(0, 8, f"- Item {idx+1}:")
                render_json(item, indent + 1)
        else:
            pdf.cell(indent * 5)
            pdf.multi_cell(0, 8, str(data))
    render_json(structured_json, indent=1)
    pdf.output(output_path)


def main_process(image_path):
    text = extract_text_from_image(image_path)
    doc_type = classify_document_type(text)
    print("Document Type: ", doc_type)
    structured_json = map_text_to_json(text, doc_type)
    print("Structured JSON: ", structured_json)
    # Export to JSON file
    if structured_json:
        os.makedirs('extracted_jsons', exist_ok=True)
        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_json_path = os.path.join('extracted_jsons', f'{image_filename}.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(structured_json, f, ensure_ascii=False, indent=2)
        print(f"Structured JSON exported to {output_json_path}")
        # Summarize
        summary = summarize_data(structured_json)
        summary_path = os.path.join('extracted_jsons', f'{image_filename}_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Summary exported to {summary_path}")
        # Create PDF
        pdf_path = os.path.join('extracted_jsons', f'{image_filename}.pdf')
        create_pdf_from_json(structured_json, summary, pdf_path)
        print(f"PDF exported to {pdf_path}")
    else:
        summary = None
    return text, doc_type, structured_json

def safe_json_loads(s):
    try:
        return json.loads(s)
    except Exception as e:
        print(f"Standard json.loads failed: {e}")
        try:
            return demjson3.decode(s)
        except Exception as e2:
            print(f"demjson3.decode also failed: {e2}")
            return None

# examples

print(main_process("images\prescription1.jpg"))
print(main_process("images\prescription2.png"))


