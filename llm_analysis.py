"""
LLM analysis module for interpreting emotion tracking results using Google Gemini API.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logger.warning("google-generativeai not installed")


def read_chunks_csv(csv_path: Path) -> List[Dict[str, str]]:
    """
    Read chunks CSV file and return as list of dictionaries.
    
    Args:
        csv_path: Path to the chunks CSV file
        
    Returns:
        List of dictionaries with chunk data
    """
    chunks = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append(row)
    return chunks


def format_chunks_for_prompt(chunks: List[Dict[str, str]]) -> str:
    """
    Format chunks data into a readable string for the LLM prompt.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Formatted string representation of chunks
    """
    lines = []
    for chunk in chunks:
        chunk_idx = chunk.get('chunk_index', 'N/A')
        start_time = chunk.get('start_time', 'N/A')
        end_time = chunk.get('end_time', 'N/A')
        transcript = chunk.get('transcript', '')
        arousal = chunk.get('arousal', 'N/A')
        dominance = chunk.get('dominance', 'N/A')
        valence = chunk.get('valence', 'N/A')
        
        lines.append(
            f"Chunk {chunk_idx} ({start_time}s - {end_time}s):\n"
            f"  Transcript: \"{transcript}\"\n"
            f"  Arousal: {arousal}, Dominance: {dominance}, Valence: {valence}"
        )
    
    return "\n\n".join(lines)


def create_sentiment_analysis_prompt(chunks_data: str) -> str:
    """
    Create a prompt for sentiment analysis focusing on emotional dips.
    
    Args:
        chunks_data: Formatted string of chunks data
        
    Returns:
        Complete prompt for the LLM
    """
    prompt = f"""You are analyzing emotional tracking data from a conversation. The data includes transcripts and three emotion dimensions:

- **Arousal**: Energy level (low = calm, high = excited)
- **Dominance**: Sense of control/power (low = submissive, high = dominant)
- **Valence**: Emotional positivity (low = negative, high = positive)

Here is the data:

{chunks_data}

Please provide a comprehensive sentiment analysis with the following focus:

1. **Overall Sentiment**: Describe the general emotional trajectory of the conversation.

2. **Emotional Dips Analysis** (CRITICAL FOCUS): 
   - Identify all significant dips in valence, arousal, or dominance
   - For each dip, explain:
     * What the person was saying during that dip
     * What emotional state this suggests (e.g., concern, hesitation, disappointment)
     * The potential reasons or triggers based on the transcript
     * How significant the dip is (mild concern vs. major concern)

3. **Emotional Recovery**: If there are dips followed by recovery, analyze how the person's emotional state changes and what might have caused the recovery.

4. **Key Insights**: Summarize the most important emotional patterns, especially any concerning dips that might indicate hesitation, concern, or negative sentiment.

5. **Recommendations**: Based on the emotional analysis, provide actionable insights about the person's emotional state and any areas that might need attention or follow-up.

Format your response in clear sections with headers. Be specific about timestamps and quotes from the transcript when discussing emotional dips."""
    
    return prompt


def analyze_chunks_with_gemini(
    csv_path: Path,
    api_key: str | None = None,
    model_name: str = "gemini-1.5-flash"
) -> str:
    """
    Analyze chunks CSV using Google Gemini API.
    
    Args:
        csv_path: Path to the chunks CSV file
        api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
        model_name: Name of the Gemini model to use
        
    Returns:
        LLM response as string
        
    Raises:
        ValueError: If API key is not provided and not in environment
        ImportError: If google-generativeai is not installed
    """
    logger.info(f"Starting Gemini analysis for: {csv_path}")
    
    if genai is None:
        logger.error("google-generativeai is not installed")
        raise ImportError(
            "google-generativeai is not installed. "
            "Install it with: pip install google-generativeai"
        )
    
    # Get API key
    if api_key is None:
        import os
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment")
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Either set it or pass api_key parameter."
            )
        logger.debug("Using API key from environment variable")
    else:
        logger.debug("Using API key from parameter")
    
    # Configure Gemini
    logger.info(f"Configuring Gemini with model: {model_name}")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        logger.debug("Gemini configured successfully")
    except Exception as e:
        logger.error(f"Error configuring Gemini: {str(e)}", exc_info=True)
        raise
    
    # Read and format chunks
    logger.info("Reading chunks CSV...")
    try:
        chunks = read_chunks_csv(csv_path)
        logger.info(f"Read {len(chunks)} chunks from CSV")
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}", exc_info=True)
        raise
    
    logger.debug("Formatting chunks for prompt...")
    chunks_data = format_chunks_for_prompt(chunks)
    logger.debug(f"Formatted chunks data length: {len(chunks_data)} characters")
    
    logger.debug("Creating sentiment analysis prompt...")
    prompt = create_sentiment_analysis_prompt(chunks_data)
    logger.debug(f"Prompt length: {len(prompt)} characters")
    
    # Generate response
    logger.info("Calling Gemini API to generate analysis...")
    try:
        response = model.generate_content(prompt)
        logger.info("Received response from Gemini API")
        
        if not response.text:
            logger.warning("Empty response from Gemini API")
            return "No analysis generated."
        
        logger.info(f"Analysis length: {len(response.text)} characters")
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}", exc_info=True)
        raise

