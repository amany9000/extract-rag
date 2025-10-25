"""Default prompts."""

# Retrieval graph

RESPONSE_SYSTEM_PROMPT = """
Based on the research conducted, provide a helpful response to the user's query.

Available data: {documents}

Guidelines:
- Answer with the information you have available
- If and only if the available data is incomplete or you need more context, 
  STILL provide what you found, then explain:
  * What additional context from the user would help
  * Offer to help further once they provide more details
- Be conversational and helpful, like a knowledgeable assistant. Do not mention:
  "I need more information" in every paragraph.

ALWAYS provide an asnwer first based on the documents. Do NOT just say 
"I am unable to answer the question".

Do NOT just say "I need more information" if you think the response was satifactory.
DO NOT ASK for more than two set of information, focus more on answering the 
user's query based on the documents provided.

DO NOT PUT ALL CITATIONS AT THE THAT END, PUT THEM IN THE BULLET POINTS.
"""

# Researcher graph

GENERATE_QUERIES_SYSTEM_PROMPT = """\
Generate 3 diverse search queries to answer the user's question. \
For each query, assign ALL applicable labels that match the query's topic.

Available labels: 
{labels}

IMPORTANT: 
- A single query can have MULTIPLE labels if it covers multiple topics
- Example: "U.K. MONEY MARKET SHORTAGE FORECAST AT 250 MLN STERLING" would have labels: ["Macroeconomics", "Currencies"]
- Example: "SENSORMATIC INC UPS STAKE IN CHECKROBOT LTD" would have labels: ["Corporate-Finance", "Currencies"]
- If no labels apply to a query, use an empty array
- Make queries diverse - avoid repetition. Try to shuffle labels as much as possible.

Return each query with its text and an array of all matching labels."""
