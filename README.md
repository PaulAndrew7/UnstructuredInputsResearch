

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


### Running the Flask API

1. **Place your images** in the `images/` directory (optional if you will upload via API)
2. **Start the server**
   ```bash
   python main.py
   ```

The server will start on `http://localhost:5000`.

### API Endpoints

- **GET `/health`**
  - Returns server status.
  - Example:
    ```bash
    curl http://localhost:5000/health
    ```

- **POST `/process`**
  - Process a single image.
  - Send either a multipart file under key `file` OR a JSON body with `image_path` pointing to an existing file on the server.
  - Examples:
    ```bash
    # Upload a file
    curl -X POST http://localhost:5000/process -F "file=@images/prescription1.jpg"

    # Or reference an existing path on the server
    curl -X POST http://localhost:5000/process -H "Content-Type: application/json" \
      -d '{"image_path": "images/prescription1.jpg"}'
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

### Deployment

You can run the app and Ollama together on a VM using Docker Compose, or point the app to a remote Ollama server.

Environment variables (optional):
- `OLLAMA_API_BASE` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `mistral`)
 - `FRONTEND_ORIGIN` (for CORS; e.g., `https://yourpage.github.io`)

Example (pointing to a remote Ollama):
```bash
set OLLAMA_API_BASE=http://<ollama-host>:11434  # Windows CMD
set OLLAMA_MODEL=mistral
python main.py
```

#### Docker (app only)
```bash
docker build -t ocr-app .
docker run -p 5000:5000 -e OLLAMA_API_BASE=http://host.docker.internal:11434 -e OLLAMA_MODEL=mistral ocr-app
```

#### Docker Compose (app + Ollama)
Use the provided `docker-compose.yml` to run both services. This pulls Ollama, starts the API, and runs the app configured to call `http://ollama:11434`.

```bash
docker compose up --build
```

Then open `http://localhost:5000`.

### Using a static frontend (e.g., GitHub Pages)
- This backend can be called from a static site via fetch/XHR over HTTPS.
- Set `FRONTEND_ORIGIN` to your static site origin (e.g., `https://<user>.github.io`) to restrict CORS in production.
- Example fetch from a static site:
```js
fetch('https://<your-public-backend-url>/process', {
  method: 'POST',
  body: (() => { const f = new FormData(); f.append('file', fileInput.files[0]); return f; })()
})
  .then(r => r.json())
  .then(console.log)
```

