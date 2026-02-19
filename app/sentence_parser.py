"""
app/sentence_parser.py — Extract symptoms from user sentences

Handles:
  - "I have a fever, headache, and vomiting"
  - "My stomach hurts. I've been throwing up all night."
  - "fever, chills, cough"
  - Multi-sentence paragraphs

Pipeline:
  User text → split sentences → extract symptom phrases → deduplicate → return list
"""

import re
from typing import List, Set

# ─────────────────────────────────────────────────────────────
# EXTRACTION PATTERNS
# These regex patterns identify symptom mentions in natural text
# ─────────────────────────────────────────────────────────────

# Common sentence structures that contain symptoms
SYMPTOM_PATTERNS = [
    # "I have X" / "I've been having X" / "I am experiencing X"
    r"(?:i\s+(?:have|had|got|get|am\s+having|'ve\s+(?:been\s+)?having|am\s+experiencing|'m\s+experiencing|suffer(?:ing)?\s+from))\s+(?:a\s+)?([^.,;]+)",
    
    # "X hurts" / "X aches" / "X is painful"
    r"(?:my\s+)?([a-z\s]+)\s+(?:hurts?|aches?|is\s+(?:hurting|aching|painful|sore|tender))",
    
    # "feeling X" / "feel X"
    r"(?:feeling|feel(?:ing)?)\s+(?:very\s+)?([^.,;]+)",
    
    # "experiencing X" / "suffering from X"
    r"(?:experiencing|suffering\s+from)\s+([^.,;]+)",
    
    # "there is X" / "there's X"
    r"(?:there\s+is|there's)\s+([^.,;]+)",
    
    # "been X" / "been having X"
    r"(?:been|having)\s+([^.,;]+)",
    
    # Direct symptom mentions after punctuation
    r"[.,;]\s*([a-z][a-z\s]+?)(?:\s+and\s+|\s*,\s*|\s*$)",
]

# Filler words to strip from extracted phrases
FILLER_WORDS = {
    "a", "an", "the", "some", "my", "very", "really", "quite", "extremely",
    "severe", "mild", "slight", "bad", "terrible", "awful", "horrible",
    "lot", "lots", "much", "many", "bit", "little", "for", "since", "about",
    "around", "over", "more", "less", "too", "so", "just", "also", "even",
    "still", "now", "right", "currently", "recently", "lately", "today",
    "tonight", "yesterday", "morning", "afternoon", "evening", "night",
    "day", "days", "week", "weeks", "month", "months"
}

# Words that indicate negation — exclude phrases with these
NEGATION_WORDS = {
    "no", "not", "never", "without", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"
}

# Common symptom keywords that help identify valid extractions
SYMPTOM_KEYWORDS = {
    "pain", "ache", "fever", "cough", "vomit", "nausea", "dizzy", "tired",
    "headache", "sore", "itch", "rash", "swell", "bleed", "burn", "cramp",
    "hurt", "weak", "numb", "tingle", "throb", "stiff", "tender", "sore",
    "discharge", "leak", "drip", "ooze", "pus", "blood", "spot", "bump",
    "lump", "mass", "growth", "lesion", "ulcer", "wound", "cut", "bruise",
    "swelling", "inflammation", "infection", "congestion", "pressure",
    "tightness", "heaviness", "fullness", "discomfort", "irritation",
    "sensitivity", "numbness", "tingling", "burning", "stinging", "throbbing",
    "pounding", "racing", "fluttering", "skipping", "irregular", "rapid",
    "slow", "fast", "high", "low", "short", "long", "frequent", "constant",
    "continuous", "persistent", "chronic", "acute", "sudden", "gradual",
    "progressive", "worsening", "improving", "recurring", "intermittent"
}


# ─────────────────────────────────────────────────────────────
# CORE PARSER CLASS
# ─────────────────────────────────────────────────────────────
class SentenceParser:
    """
    Extract symptom phrases from natural language text.
    
    Handles multi-sentence input, lists, and conversational descriptions.
    """
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in SYMPTOM_PATTERNS]
    
    def _normalize(self, text: str) -> str:
        """Basic text cleanup."""
        text = text.lower().strip()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize punctuation
        text = re.sub(r'[!?]', '.', text)
        return text
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving commas within lists."""
        # Split on periods, but not on abbreviations
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_from_list(self, text: str) -> List[str]:
        """
        Extract symptoms from comma/and-separated lists.
        Example: "fever, chills, and vomiting" → ["fever", "chills", "vomiting"]
        """
        # Remove leading conjunctions
        text = re.sub(r'^(?:and|or)\s+', '', text, flags=re.IGNORECASE)
        
        # Split on commas and "and"/"or"
        parts = re.split(r'\s*(?:,|and|or)\s*', text, flags=re.IGNORECASE)
        
        # Clean each part
        symptoms = []
        for part in parts:
            part = part.strip()
            if not part or len(part) < 3:
                continue
            
            # Remove filler words
            words = part.split()
            filtered = [w for w in words if w.lower() not in FILLER_WORDS]
            if filtered:
                clean = ' '.join(filtered)
                if self._looks_like_symptom(clean):
                    symptoms.append(clean)
        
        return symptoms
    
    def _has_negation(self, text: str) -> bool:
        """Check if text contains negation words."""
        words = set(text.lower().split())
        return bool(words & NEGATION_WORDS)
    
    def _looks_like_symptom(self, text: str) -> bool:
        """
        Heuristic check: does this phrase look like a symptom?
        - Contains symptom keywords
        - Is reasonable length (2-30 chars)
        - Not just filler words
        """
        if len(text) < 2 or len(text) > 30:
            return False
        
        words = set(text.lower().split())
        
        # All filler words? Not a symptom
        if words <= FILLER_WORDS:
            return False
        
        # Contains symptom keyword? Likely a symptom
        if any(kw in text.lower() for kw in SYMPTOM_KEYWORDS):
            return True
        
        # Short and non-filler? Probably a symptom
        if len(words) <= 3 and words - FILLER_WORDS:
            return True
        
        return False
    
    def _clean_extracted_phrase(self, phrase: str) -> str:
        """Final cleanup of extracted phrase."""
        # Strip leading/trailing filler words
        words = phrase.split()
        while words and words[0].lower() in FILLER_WORDS:
            words.pop(0)
        while words and words[-1].lower() in FILLER_WORDS:
            words.pop()
        
        phrase = ' '.join(words)
        
        # Remove trailing punctuation except hyphens/underscores
        phrase = re.sub(r'[.,;:!?]+$', '', phrase)
        
        return phrase.strip()
    
    def extract_symptoms(self, text: str) -> List[str]:
        """
        Main method: extract symptom phrases from natural language text.
        
        Returns:
            List of symptom phrases (may contain duplicates or near-duplicates)
        """
        if not text or not text.strip():
            return []
        
        text = self._normalize(text)
        symptoms: Set[str] = set()
        
        # ── Strategy 1: Pattern matching ──
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                captured = match.group(1).strip()
                
                # Skip if negated
                full_context = text[max(0, match.start()-20):match.end()]
                if self._has_negation(full_context):
                    continue
                
                # Check if it's a list
                if ',' in captured or ' and ' in captured:
                    symptoms.update(self._extract_from_list(captured))
                else:
                    clean = self._clean_extracted_phrase(captured)
                    if clean and self._looks_like_symptom(clean):
                        symptoms.add(clean)
        
        # ── Strategy 2: Comma-separated lists at start ──
        # "fever, chills, vomiting" without any verb
        if ',' in text and len(text) < 200:  # short input likely a list
            first_sentence = text.split('.')[0]
            if first_sentence.count(',') >= 1:
                list_symptoms = self._extract_from_list(first_sentence)
                symptoms.update(list_symptoms)
        
        # ── Strategy 3: Single-word symptoms ──
        # If input is very short (1-3 words), treat it as a symptom directly
        words = text.split()
        if len(words) <= 3 and not any(c in text for c in '.,;'):
            clean = self._clean_extracted_phrase(text)
            if clean and self._looks_like_symptom(clean):
                symptoms.add(clean)
        
        return list(symptoms)
    
    def parse(self, text: str) -> dict:
        """
        Parse text and return detailed results.
        
        Returns:
            {
                "input": original text,
                "extracted": list of symptom phrases,
                "count": number of symptoms found
            }
        """
        symptoms = self.extract_symptoms(text)
        
        return {
            "input": text.strip(),
            "extracted": symptoms,
            "count": len(symptoms)
        }


# ─────────────────────────────────────────────────────────────
# SINGLETON — import this in main.py
# ─────────────────────────────────────────────────────────────
parser = SentenceParser()