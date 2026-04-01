import os
import re
import unicodedata

def normalize_vietnamese(text: str) -> str:
    """
    Normalize Vietnamese transcripts before tokenization.
    - lowercase
    - unicode normalization
    - strip extra spaces
    - remove repeated punctuation
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    # Unicode normalization (NFC is standard for Vietnamese)
    text = unicodedata.normalize('NFC', text)
    # Remove repeated punctuation
    text = re.sub(r'([^\w\s])\1+', r'\1', text)
    # Strip extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def test_normalize():
    input_text = "Xin chào  mọi người!!!"
    output = normalize_vietnamese(input_text)
    assert output == "xin chào mọi người", f"Got: '{output}'"
    print("Normalization test passed.")

if __name__ == "__main__":
    test_normalize()
