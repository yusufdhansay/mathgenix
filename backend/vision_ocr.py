"""
vision_ocr.py — Groq Vision (Llama 4 Scout) text extraction for MathGenix.

Uses Groq's multimodal Llama 4 Scout model to extract text from images
of study materials (phone photos, scanned pages, handwritten notes).
Handles auto-compression for large images and HEIC (iPhone) conversion.
"""

import io
import base64
import os
import requests
from PIL import Image

# Maximum base64 payload size Groq allows (4 MB)
MAX_BASE64_BYTES = 4 * 1024 * 1024

# Target JPEG quality for compression
COMPRESS_QUALITY = 85

# Maximum dimension (width or height) before downscaling
MAX_DIMENSION = 2048

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Image extensions we support
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.bmp', '.tiff', '.tif'}


def _load_image(image_bytes: bytes, filename: str) -> Image.Image:
    """
    Loads image bytes into a Pillow Image object.
    Handles HEIC/HEIF (iPhone format) by attempting pillow-heif import.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in ('.heic', '.heif'):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            raise ValueError(
                "HEIC/HEIF support requires the 'pillow-heif' package. "
                "Install it with: pip install pillow-heif"
            )

    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert palette / RGBA images to RGB for JPEG encoding
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        raise ValueError(f"Failed to open image file '{filename}': {str(e)}")


def _compress_to_base64(img: Image.Image) -> str:
    """
    Compresses and encodes a Pillow Image to a base64 JPEG string.
    Downscales if the image is too large, and reduces quality iteratively
    until the base64 payload fits within Groq's 4 MB limit.
    """
    # Downscale if dimensions are very large
    w, h = img.size
    if max(w, h) > MAX_DIMENSION:
        ratio = MAX_DIMENSION / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f">>> [VISION] Resized image from {w}x{h} to {new_w}x{new_h}")

    quality = COMPRESS_QUALITY
    while quality >= 20:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        jpeg_bytes = buffer.getvalue()
        b64_str = base64.b64encode(jpeg_bytes).decode('utf-8')

        # Check if base64 payload is within limits
        if len(b64_str) <= MAX_BASE64_BYTES:
            size_kb = len(jpeg_bytes) / 1024
            print(f">>> [VISION] Compressed to {size_kb:.0f} KB (quality={quality})")
            return b64_str

        # Reduce quality and try again
        quality -= 10
        print(f">>> [VISION] Image too large at quality={quality + 10}, retrying at {quality}...")

    # Last resort: aggressive downscale
    w, h = img.size
    img = img.resize((w // 2, h // 2), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=40, optimize=True)
    b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print(f">>> [VISION] Emergency downscale to {w // 2}x{h // 2}")
    return b64_str


def extract_text_from_image(image_bytes: bytes, filename: str, api_key: str) -> str:
    """
    Extracts text from a single image using Groq's Llama 4 Scout vision model.

    Args:
        image_bytes: Raw bytes of the image file
        filename: Original filename (used for extension detection)
        api_key: Groq API key

    Returns:
        Extracted text content from the image

    Raises:
        ValueError: If the image cannot be processed or API call fails
    """
    if not api_key or api_key == "your_key_here":
        raise ValueError(
            "Groq API key is required for image text extraction. "
            "Please configure your API key in Settings."
        )

    # Load and compress
    img = _load_image(image_bytes, filename)
    b64_image = _compress_to_base64(img)

    print(f">>> [VISION] Sending image '{filename}' to Llama 4 Scout for text extraction...")

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL text from this image exactly as written. "
                            "Preserve mathematical notation, equations, formulas, symbols, "
                            "and any special characters. If there are tables, preserve their "
                            "structure using pipes (|) and dashes. If there are numbered lists "
                            "or bullet points, preserve them. Output ONLY the extracted text "
                            "content — no descriptions, no commentary, no formatting instructions."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,  # Vision requests can take longer
        )

        if response.status_code == 401:
            raise ValueError("Invalid Groq API key for vision extraction.")
        elif response.status_code == 429:
            raise ValueError(
                "Groq rate limit exceeded. Please wait a moment before uploading again "
                "(free tier: 30K tokens/min)."
            )
        elif response.status_code != 200:
            raise ValueError(
                f"Groq Vision API error ({response.status_code}): "
                f"{response.text[:300]}"
            )

        result = response.json()
        extracted_text = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        usage = result.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        print(f">>> [VISION] Extraction complete: {len(extracted_text)} chars, {tokens} tokens used")

        if not extracted_text or not extracted_text.strip():
            raise ValueError(
                "The vision model could not extract any text from this image. "
                "Please ensure the image contains readable text or mathematical content."
            )

        return extracted_text.strip()

    except requests.exceptions.Timeout:
        raise ValueError("Vision extraction timed out after 60 seconds. Please try again.")
    except requests.exceptions.ConnectionError:
        raise ValueError("Cannot connect to Groq API. Please check your internet connection.")
    except ValueError:
        raise  # Re-raise our own ValueErrors
    except Exception as e:
        raise ValueError(f"Vision text extraction failed: {str(e)}")


def extract_text_from_pdf_page_image(page, page_num: int, api_key: str) -> str:
    """
    Renders a single PDF page to an image and extracts text via vision OCR.
    Used as a fallback when pypdf cannot extract text (scanned/image-only PDFs).

    Args:
        page: A pypdf page object
        page_num: Page number (for logging)
        api_key: Groq API key

    Returns:
        Extracted text from the page image
    """
    try:
        # Extract images embedded in the PDF page
        images = page.images
        if not images:
            return ""

        # Use the largest image on the page (typically the scanned page itself)
        largest_image = max(images, key=lambda img: len(img.data))
        image_bytes = largest_image.data

        print(f">>> [VISION] PDF page {page_num + 1}: extracted embedded image "
              f"({len(image_bytes) / 1024:.0f} KB), sending to vision OCR...")

        return extract_text_from_image(
            image_bytes,
            f"page_{page_num + 1}.jpg",
            api_key
        )
    except ValueError:
        raise  # Re-raise API/auth errors
    except Exception as e:
        print(f">>> [VISION] PDF page {page_num + 1} image extraction failed: {e}")
        return ""
