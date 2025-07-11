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

