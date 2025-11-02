import re

class QueryFilter:
    """Filter to determine if a query is document-related."""
    
    CASUAL_GREETINGS = [
        "hi", "hello", "hey", "how are you", "what's up", "whats up",
        "good morning", "good afternoon", "good evening", "how do you do",
        "nice to meet you", "how's it going", "what's going on", "sup",
        "yo", "hiya", "howdy", "greetings", "salutations"
    ]
    
    @classmethod
    def is_document_related_query(cls, query: str) -> bool:
        """Check if the query is document-related."""
        query_lower = query.lower().strip()
        query_clean = re.sub(r'[^\w\s]', '', query_lower)
        
        # Check for casual greetings
        for greeting in cls.CASUAL_GREETINGS:
            if query_clean == greeting or query_clean.startswith(greeting + " "):
                return False
        
        # Check for very short queries (likely not document-related)
        if len(query_clean.split()) <= 2 and any(word in query_clean for word in ["hi", "hey", "hello", "sup", "yo"]):
            return False
        
        # If it contains question words or seems like a meaningful query, consider it document-related
        question_indicators = ["what", "how", "when", "where", "why", "who", "which", "explain", "describe", "tell", "show"]
        if any(indicator in query_lower for indicator in question_indicators):
            return True
        
        # If it's longer than 3 words and doesn't match casual patterns, likely document-related
        if len(query_clean.split()) > 3:
            return True
        
        return False