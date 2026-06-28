import json
from openai import OpenAI

# Connect to local Ollama or LM Studio
LOCAL_LLM_BASE_URL = "http://localhost:11434/v1"
LOCAL_LLM_API_KEY = "local-reasoning"

def get_first_available_model(client: OpenAI) -> str:
    """Dynamically grab the first model you have installed in Ollama to avoid hardcoding names!"""
    try:
        models = client.models.list().data
        if models:
            return models[0].id
    except Exception:
        pass
    return "llama3"

def parse_ocr_text_with_reasoning(raw_text: str) -> dict | None:
    """
    Feeds the perfectly accurate TrOCR text into a Local Reasoning LLM
    so it can intelligently categorize it into Dimensions, GD&T, and Notes.
    """
    if not raw_text.strip():
        return None
        
    try:
        client = OpenAI(base_url=LOCAL_LLM_BASE_URL, api_key=LOCAL_LLM_API_KEY, timeout=30.0)
        model = get_first_available_model(client)
        
        system_prompt = (
            "You are an expert mechanical engineering AI. Your job is to parse raw OCR text from an engineering drawing "
            "and output a strictly formatted JSON object categorizing the text. "
            "Do not output markdown, only valid JSON.\n\n"
            "Format exactly like this:\n"
            "{\n"
            "  \"dimensions\": [\"list of length/diameter dimensions\"],\n"
            "  \"gdt_frames\": [\"list of Geometric Dimensioning and Tolerancing frames (e.g. 0.05 B)\"],\n"
            "  \"manufacturing_notes\": [\"list of general handling, finishing, or manufacturing notes\"],\n"
            "  \"materials\": [\"list of material specs\"]\n"
            "}"
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Parse this OCR text into JSON:\n\n{raw_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"\n[!] WARNING: Local Reasoning LLM failed to parse. Is Ollama/LM Studio running on {LOCAL_LLM_BASE_URL}?\nError: {e}\n")
        return None
