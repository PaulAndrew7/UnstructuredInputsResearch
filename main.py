import json
import os
import pytesseract
from PIL import Image
import cv2
import numpy as np
import requests
import demjson3
import glob
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS

# Deployment/config
OLLAMA_API_BASE = os.environ.get('OLLAMA_API_BASE', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'mistral')
from werkzeug.utils import secure_filename


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
    os.makedirs('text_extract', exist_ok=True)
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
Always include a field called \"document_type\" with the value \"{doc_type}\".

Text:
{text}

Return only the JSON object.
"""
    print("=== PROMPT SENT TO MODEL ===")
    print(prompt)
    response = requests.post(
        f"{OLLAMA_API_BASE}/api/generate",
        json={
            "model": OLLAMA_MODEL,  # or "phi", "llama2", etc.
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
        json_str = match.group(1)
        # Replace curly quotes and other common LLM output issues
        json_str = json_str.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
        return safe_json_loads(json_str)
    else:
        return None

def safe_json_loads(s):
    try:
        return json.loads(s)
    except Exception as e:
        print(f"Standard json.loads failed: {e}")
        try:
            return demjson3.decode(s)
        except Exception as e2:
            print(f"demjson3.decode also failed: {e2}")
            print(f"Raw JSON string that failed to parse: {s}")
            # Try to fix common issues
            try:
                # Remove any trailing commas before closing braces/brackets
                import re
                fixed_s = re.sub(r',(\s*[}\]])', r'\1', s)
                # Try to fix missing quotes around keys
                fixed_s = re.sub(r'(\w+):', r'"\1":', fixed_s)
                return json.loads(fixed_s)
            except Exception as e3:
                print(f"Attempted fix also failed: {e3}")
                return None
    return None

def fallback_extract_fields(text, doc_type):
    """Fallback extraction using regex patterns if JSON parsing fails"""
    import re
    result = {"document_type": doc_type}
    
    # Extract common patterns with more comprehensive matching
    patient_patterns = [
        r'patient[:\s]+([^\n]+)',
        r'name[:\s]+([^\n]+)',
        r'patient\s+name[:\s]+([^\n]+)'
    ]
    for pattern in patient_patterns:
        patient_match = re.search(pattern, text, re.IGNORECASE)
        if patient_match:
            result["patient_name"] = patient_match.group(1).strip()
            break
    
    doctor_patterns = [
        r'(?:doctor|dr)[:\s]+([^\n]+)',
        r'physician[:\s]+([^\n]+)',
        r'prescriber[:\s]+([^\n]+)'
    ]
    for pattern in doctor_patterns:
        doctor_match = re.search(pattern, text, re.IGNORECASE)
        if doctor_match:
            result["doctor_name"] = doctor_match.group(1).strip()
            break
    
    date_patterns = [
        r'date[:\s]+([^\n]+)',
        r'prescribed[:\s]+([^\n]+)',
        r'issued[:\s]+([^\n]+)'
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, text, re.IGNORECASE)
        if date_match:
            result["date"] = date_match.group(1).strip()
            break
    
    # Extract medications with more detail
    medications = []
    
    # Look for medication patterns
    med_patterns = [
        r'(\w+(?:\s+\w+)*\s+\d+mg[^\n]*)',
        r'(\w+(?:\s+\w+)*\s+tablet[^\n]*)',
        r'(\w+(?:\s+\w+)*\s+capsule[^\n]*)',
        r'(\w+(?:\s+\w+)*\s+\d+\s*mg[^\n]*)',
        r'rx[:\s]+([^\n]+)',
        r'prescription[:\s]+([^\n]+)'
    ]
    
    for pattern in med_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            med_info = {"name": match.strip()}
            
            # Try to extract dosage from the same line
            dosage_match = re.search(r'(\d+\s*mg)', match, re.IGNORECASE)
            if dosage_match:
                med_info["dosage"] = dosage_match.group(1)
            
            # Try to extract frequency
            freq_patterns = [
                r'(\d+\s+times?\s+(?:a\s+)?day)',
                r'(once\s+(?:a\s+)?day)',
                r'(twice\s+(?:a\s+)?day)',
                r'(three\s+times?\s+(?:a\s+)?day)',
                r'(bid|tid|qid|od|bd|hs)',
                r'(\d+\s+times?\s+daily)'
            ]
            for freq_pattern in freq_patterns:
                freq_match = re.search(freq_pattern, match, re.IGNORECASE)
                if freq_match:
                    med_info["frequency"] = freq_match.group(1)
                    break
            
            # Try to extract duration
            duration_match = re.search(r'for\s+(\d+\s+(?:days?|weeks?|months?))', match, re.IGNORECASE)
            if duration_match:
                med_info["duration"] = duration_match.group(1)
            
            medications.append(med_info)
    
    if medications:
        result["medications"] = medications
    
    # Extract instructions
    instruction_patterns = [
        r'instructions[:\s]+([^\n]+)',
        r'sig[:\s]+([^\n]+)',
        r'take[:\s]+([^\n]+)',
        r'usage[:\s]+([^\n]+)'
    ]
    for pattern in instruction_patterns:
        instruction_match = re.search(pattern, text, re.IGNORECASE)
        if instruction_match:
            result["instructions"] = instruction_match.group(1).strip()
            break
    
    # Extract pharmacy information
    pharmacy_match = re.search(r'pharmacy[:\s]+([^\n]+)', text, re.IGNORECASE)
    if pharmacy_match:
        result["pharmacy"] = pharmacy_match.group(1).strip()
    
    # Extract refill information
    refill_match = re.search(r'refill[:\s]+([^\n]+)', text, re.IGNORECASE)
    if refill_match:
        result["refill_info"] = refill_match.group(1).strip()
    
    # If we found very little, try to extract any line that looks like it contains data
    if len(result) <= 2:  # Only document_type and maybe one other field
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line and len(line) > 5:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip()
                    if key not in result and value:
                        result[key] = value
    
    return result

def map_text_to_json(text, doc_type):
    ai_result = ollama_map_text_to_json(text, doc_type)
    if ai_result:
        if 'document_type' not in ai_result:
            ai_result['document_type'] = doc_type
        return ai_result
    else:
        print("AI extraction failed, trying fallback extraction...")
        return fallback_extract_fields(text, doc_type)

def summarize_data(structured_json):
    import requests
    prompt = f"""
Summarize the following structured data in 100-150 words, highlighting the key content, topics covered, or instructions mentioned. Write in clear, professional language.

Data:
{json.dumps(structured_json, indent=2)}
"""
    response = requests.post(
        f"{OLLAMA_API_BASE}/api/generate",
        json={
            "model": OLLAMA_MODEL,  # or "phi", "llama2", etc.
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
    pdf.cell(0, 10, f"Document Type: {structured_json.get('document_type', 'Unknown').capitalize()}", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"{summary}")
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Extracted Data:", ln=True)
    pdf.set_font("Arial", '', 12)
    
    def flatten_json(data, prefix=""):
        """Flatten nested JSON into key-value pairs"""
        items = []
        if isinstance(data, dict):
            for k, v in data.items():
                if k == 'document_type':  # Skip document_type as it's already in header
                    continue
                if isinstance(v, (dict, list)):
                    items.extend(flatten_json(v, f"{prefix}{k}_"))
                else:
                    items.append((f"{prefix}{k}", str(v)))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    items.extend(flatten_json(item, f"{prefix}item_{i+1}_"))
                else:
                    items.append((f"{prefix}item_{i+1}", str(item)))
        return items
    
    # Get flattened data
    flat_data = flatten_json(structured_json)
    
    # Create table
    col_width = [60, 120]  # Field name width, Value width
    row_height = 8
    
    # Table header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(col_width[0], row_height, "Field", border=1)
    pdf.cell(col_width[1], row_height, "Value", border=1, ln=True)
    
    # Table data
    pdf.set_font("Arial", '', 10)
    for field, value in flat_data:
        # Handle long values by wrapping text
        if len(value) > 50:  # If value is long, use multi_cell
            pdf.cell(col_width[0], row_height, field.replace('_', ' ').title(), border=1)
            # Calculate how many lines this value will take
            lines = len(value) // 50 + 1
            pdf.multi_cell(col_width[1], row_height, value, border=1)
        else:
            pdf.cell(col_width[0], row_height, field.replace('_', ' ').title(), border=1)
            pdf.cell(col_width[1], row_height, value, border=1, ln=True)
    
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
        try:
            summary = summarize_data(structured_json)
            if summary:
                summary_path = os.path.join('extracted_jsons', f'{image_filename}_summary.txt')
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"Summary exported to {summary_path}")
            else:
                print("Warning: Summary generation failed")
                summary = "Summary generation failed"
        except Exception as e:
            print(f"Error generating summary: {e}")
            summary = "Summary generation failed"
        # Create PDF
        try:
            pdf_path = os.path.join('extracted_jsons', f'{image_filename}.pdf')
            create_pdf_from_json(structured_json, summary, pdf_path)
            print(f"PDF exported to {pdf_path}")
        except Exception as e:
            print(f"Error creating PDF: {e}")
    else:
        summary = None
    return text, doc_type, structured_json

app = Flask(__name__)
FRONTEND_ORIGIN = os.environ.get('FRONTEND_ORIGIN')
if FRONTEND_ORIGIN:
    CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})
else:
    # Default: allow all origins for development; set FRONTEND_ORIGIN in production
    CORS(app)

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def save_uploaded_file(uploaded_file):
    ensure_directory('uploads')
    filename = secure_filename(uploaded_file.filename)
    saved_path = os.path.join('uploads', filename)
    uploaded_file.save(saved_path)
    return saved_path

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_single():
    # Accept either multipart file upload under key 'file' or a JSON body with 'image_path'
    image_path = None

    if 'file' in request.files and request.files['file']:
        uploaded_file = request.files['file']
        if uploaded_file.filename:
            image_path = save_uploaded_file(uploaded_file)
    elif request.is_json:
        body = request.get_json(silent=True) or {}
        image_path = body.get('image_path')
    else:
        image_path = request.form.get('image_path')

    if not image_path or not os.path.exists(image_path):
        return jsonify({
            'error': 'No valid image provided. Upload a file under key "file" or provide an existing path in "image_path".'
        }), 400

    try:
        text, doc_type, structured_json = main_process(image_path)
        response_payload = {
            'image_path': image_path,
            'document_type': doc_type,
            'text': text,
            'structured_json': structured_json
        }
        # Build file outputs for website result page
        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        json_name = f'{image_filename}.json'
        pdf_name = f'{image_filename}.pdf'
        summary_name = f'{image_filename}_summary.txt'
        return_fmt = request.headers.get('Accept', '')
        if 'text/html' in return_fmt or request.args.get('html') == '1' or 'file' in request.files:
            return render_template(
                'result.html',
                image_uploaded=('file' in request.files),
                uploaded_filename=os.path.basename(image_path) if ('file' in request.files) else None,
                image_path=image_path,
                document_type=doc_type,
                text=text,
                structured_json=structured_json,
                json_name=json_name,
                pdf_name=pdf_name,
                summary_name=summary_name
            )
        return jsonify(response_payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/files/<path:filename>')
def serve_generated_file(filename):
    return send_from_directory('extracted_jsons', filename)

@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/browse', methods=['GET'])
def browse_outputs():
    ensure_directory('extracted_jsons')
    files = sorted(os.listdir('extracted_jsons'))
    # Group by base name
    grouped = {}
    for name in files:
        base, ext = os.path.splitext(name)
        grouped.setdefault(base, []).append(name)
    items = []
    for base, names in grouped.items():
        items.append({
            'base': base,
            'json': f'{base}.json' if f'{base}.json' in names else None,
            'pdf': f'{base}.pdf' if f'{base}.pdf' in names else None,
            'summary': f'{base}_summary.txt' if f'{base}_summary.txt' in names else None
        })
    return render_template('browse.html', items=sorted(items, key=lambda x: x['base']))

if __name__ == '__main__':
    # Start the Flask app instead of running the CLI loop
    app.run(host='0.0.0.0', port=5000)