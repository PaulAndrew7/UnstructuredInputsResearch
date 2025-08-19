

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

4. **Install and setup Ollama**
   ```bash
   # Download Ollama from https://ollama.com/
   # Pull the Mistral model
   ollama pull mistral
   # Start Ollama server
   ollama serve
   ```


## Project Structure

```
├── main.py                 # Main processing script
├── images/                 # Input images directory
├── text_extract/          # Raw OCR text output
├── extracted_jsons/       # Generated JSON, PDF, and summary files
├── json_presets/          # JSON template structures
│   ├── prescription.json
│   └── record.json
└── README.md
```


### Basic Usage

1. **Place your images** in the `images/` directory
2. **Run the processing script**
   ```bash
   python main.py
   ```

### Supported Image Formats
- JPG/JPEG
- PNG
- BMP
- TIFF

### Output Files

For each processed image, the system generates:
- `{filename}.json` - Structured data in JSON format
- `{filename}.pdf` - Professional PDF with tabular data
- `{filename}_summary.txt` - Content summary
