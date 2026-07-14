"""
Gemini Vision integration for AutoQA.
Sends screenshots to Gemini and asks it to verify what's on screen
against the expected state from the test description.
"""

import os
import base64
import json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def verify_screenshot(
    screenshot_path: str,
    expected_description: str,
    question: str = None,
) -> dict:
    """
    Send a screenshot to Gemini Vision and ask it to verify
    whether the screen matches the expected description.

    Returns:
        {
          "verified": bool,
          "confidence": "high|medium|low",
          "description": "what Gemini sees on screen",
          "matches": "explanation of match/mismatch",
          "suggestion": "what might be wrong if not matching"
        }
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — skipping vision verification")
        return {
            "verified": None,
            "confidence": "none",
            "description": "Vision check skipped (no API key)",
            "matches": "N/A",
            "suggestion": "Add GEMINI_API_KEY to enable visual verification",
        }

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        with open(screenshot_path, "rb") as f:
            img_bytes = f.read()

        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""You are a QA engineer verifying a browser screenshot.

EXPECTED STATE: {expected_description}
{f"SPECIFIC QUESTION: {question}" if question else ""}

Look at this screenshot and answer:
1. What do you see on the screen? (brief description)
2. Does it match the expected state?
3. If not, what seems wrong?

Respond ONLY with valid JSON:
{{
  "verified": true/false,
  "confidence": "high/medium/low",
  "description": "what you see on screen in 1-2 sentences",
  "matches": "explanation of whether/why it matches or doesn't",
  "suggestion": "if not verified, what might fix it (or 'N/A' if verified)"
}}"""

        import PIL.Image
        import io
        img = PIL.Image.open(io.BytesIO(img_bytes))
        response = model.generate_content([prompt, img])

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        logger.info(f"Vision check: verified={result.get('verified')} confidence={result.get('confidence')}")
        return result

    except ImportError:
        return {
            "verified": None,
            "confidence": "none",
            "description": "google-generativeai not installed",
            "matches": "N/A",
            "suggestion": "pip install google-generativeai pillow",
        }
    except Exception as e:
        logger.warning(f"Vision verification error: {e}")
        return {
            "verified": None,
            "confidence": "none",
            "description": f"Vision check failed: {str(e)}",
            "matches": "N/A",
            "suggestion": "Check GEMINI_API_KEY and screenshot path",
        }


def describe_screenshot(screenshot_path: str) -> str:
    """Just describe what's on screen — used for debugging"""
    result = verify_screenshot(
        screenshot_path,
        expected_description="anything",
        question="Describe what you see on this webpage in detail.",
    )
    return result.get("description", "Could not describe screenshot")
