import openai
import json
from src.config import OPENAI_API_KEY

# Configure the OpenAI client with your API key
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    print("Warning: OPENAI_API_KEY is not set. AI labeling will fail.")

def get_ai_cluster_analysis(terms_in_cluster, used_themes=None):
    """
    Sends a list of terms to GPT-4o and asks for a theme and marketing insight.

    Args:
        terms_in_cluster (list): A list of search term strings from a single cluster.
        used_themes (set): A set of already used themes.

    Returns:
        tuple: A tuple containing the theme (str) and insight (str).
    """
    # Use a sample of terms to keep the prompt concise and cost-effective
    sample_terms = terms_in_cluster[:25]
    term_list_str = "\n".join([f"- {term}" for term in sample_terms])
    used_themes_str = ""
    if used_themes:
        used_themes_str = (
            "Themes already used for other clusters: "
            + ", ".join(f'"{theme}"' for theme in used_themes)
            + ". Do NOT use these as the theme for this cluster."
        )

    prompt = f"""
    You are a senior marketing analyst. Given the following list of search queries from a keyword cluster, do the following:
    1. Provide a concise, unique theme label (2-4 words) that is NOT used for any other cluster in this analysis.
    2. Write a one-sentence marketing insight that starts with "Users searching these terms are likely interested in..." and clearly differentiates this cluster from others, focusing on the specific action, need, or stage in the customer journey.
    {used_themes_str}

    Search Queries:
    {term_list_str}

    Your response MUST be a valid JSON object with two keys: "theme" and "insight".
    - "theme": A 2-4 word descriptive label for this cluster's central theme.
    - "insight": A one-sentence marketing insight beginning with "Users searching these terms are likely interested in...".

    Example response:
    {{
        "theme": "Affordable Student Housing",
        "insight": "Users searching these terms are likely interested in finding budget-friendly accommodation options, often with bills included."
    }}
    """

    try:
        if not openai.api_key:
            raise ValueError("OpenAI API key is not configured.")

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful marketing analyst that provides insights in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get('theme', 'AI Theme Error'), result.get('insight', 'AI Insight Error')
    except Exception as e:
        print(f"An error occurred while getting AI analysis for cluster: {e}")
        return "Manual Label Required", "Could not generate AI insight due to an error."

def get_contextual_chat_response(user_question, cluster_summary_str):
    """
    Provides a contextual answer to a user's question based on cluster data.
    """
    prompt = f"""
    You are a senior marketing analyst. A user is asking a question about the search term cluster data they are seeing.
    Use the provided cluster summary data to form your answer. Be concise and focus on actionable marketing insights.

    CLUSTER SUMMARY DATA:
    ---
    {cluster_summary_str}
    ---

    USER QUESTION:
    "{user_question}"

    Your answer:
    """
    try:
        if not openai.api_key:
            raise ValueError("OpenAI API key is not configured.")

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful marketing analyst answering questions based on provided data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred during chat response generation: {e}")
        return "Sorry, I couldn't process your question due to an error." 