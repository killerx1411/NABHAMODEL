"""
app/symptom_mapper.py — Natural language symptom mapping layer

Converts what users actually type into the model's exact symptom vocabulary.

Pipeline:
  User input → normalize → exact match → synonym lookup → fuzzy match → suggestions

Examples:
  "fever"              → high_fever
  "I have a fever"     → high_fever
  "stomach hurts"      → stomach_pain
  "throwing up"        → vomiting
  "can't stop sneezing"→ continuous_sneezing
  "peeing a lot"       → polyuria
"""

from rapidfuzz import process, fuzz
from typing import List, Tuple, Optional
import re

# ─────────────────────────────────────────────────────────────
# FULL SYMPTOM VOCABULARY
# Must match exactly what the model was trained on
# ─────────────────────────────────────────────────────────────
SYMPTOM_VOCABULARY = [
    "abdominal_pain", "abnormal_menstruation", "acidity", "acute_liver_failure",
    "altered_sensorium", "anxiety", "back_pain", "belly_pain", "blackheads",
    "bladder_discomfort", "blister", "blood_in_sputum", "bloody_stool",
    "blurred_and_distorted_vision", "breathlessness", "brittle_nails", "bruising",
    "burning_micturition", "chest_pain", "chills", "cold_hands_and_feets", "coma",
    "congestion", "constipation", "continuous_feel_of_urine", "continuous_sneezing",
    "cough", "cramps", "dark_urine", "dehydration", "depression", "diarrhoea",
    "dischromic _patches", "distention_of_abdomen", "dizziness",
    "drying_and_tingling_lips", "enlarged_thyroid", "excessive_hunger",
    "extra_marital_contacts", "family_history", "fast_heart_rate", "fatigue",
    "fluid_overload", "foul_smell_of urine", "headache", "high_fever",
    "hip_joint_pain", "history_of_alcohol_consumption", "increased_appetite",
    "indigestion", "inflammatory_nails", "internal_itching", "irregular_sugar_level",
    "irritability", "irritation_in_anus", "itching", "joint_pain", "knee_pain",
    "lack_of_concentration", "lethargy", "loss_of_appetite", "loss_of_balance",
    "loss_of_smell", "malaise", "mild_fever", "mood_swings", "movement_stiffness",
    "mucoid_sputum", "muscle_pain", "muscle_wasting", "muscle_weakness", "nausea",
    "neck_pain", "nodal_skin_eruptions", "obesity", "pain_behind_the_eyes",
    "pain_during_bowel_movements", "pain_in_anal_region", "painful_walking",
    "palpitations", "passage_of_gases", "patches_in_throat", "phlegm", "polyuria",
    "prominent_veins_on_calf", "puffy_face_and_eyes", "pus_filled_pimples",
    "receiving_blood_transfusion", "receiving_unsterile_injections",
    "red_sore_around_nose", "red_spots_over_body", "redness_of_eyes", "restlessness",
    "runny_nose", "rusty_sputum", "scurring", "shivering", "silver_like_dusting",
    "sinus_pressure", "skin_peeling", "skin_rash", "slurred_speech",
    "small_dents_in_nails", "spinning_movements", "spotting_ urination", "stiff_neck",
    "stomach_bleeding", "stomach_pain", "sunken_eyes", "sweating",
    "swelled_lymph_nodes", "swelling_joints", "swelling_of_stomach",
    "swollen_blood_vessels", "swollen_extremeties", "swollen_legs",
    "throat_irritation", "toxic_look_(typhos)", "ulcers_on_tongue", "unsteadiness",
    "visual_disturbances", "vomiting", "watering_from_eyes", "weakness_in_limbs",
    "weakness_of_one_body_side", "weight_gain", "weight_loss", "yellow_crust_ooze",
    "yellow_urine", "yellowing_of_eyes", "yellowish_skin"
]

# ─────────────────────────────────────────────────────────────
# SYNONYM DICTIONARY
# Maps natural language phrases → exact model symptom names
# Covers the most common ways users describe symptoms
# ─────────────────────────────────────────────────────────────
SYNONYM_MAP = {
    # FEVER variants
    "fever": "high_fever",
    "high fever": "high_fever",
    "low fever": "mild_fever",
    "mild fever": "mild_fever",
    "slight fever": "mild_fever",
    "temperature": "high_fever",
    "running a temperature": "high_fever",
    "burning up": "high_fever",
    "feverish": "mild_fever",
    "pyrexia": "high_fever",

    # PAIN variants
    "stomach hurts": "stomach_pain",
    "stomach ache": "stomach_pain",
    "tummy ache": "stomach_pain",
    "abdominal pain": "abdominal_pain",
    "belly pain": "belly_pain",
    "belly ache": "belly_pain",
    "chest hurts": "chest_pain",
    "chest tightness": "chest_pain",
    "chest pressure": "chest_pain",
    "back hurts": "back_pain",
    "backache": "back_pain",
    "lower back pain": "back_pain",
    "knee hurts": "knee_pain",
    "knee ache": "knee_pain",
    "hip pain": "hip_joint_pain",
    "hip hurts": "hip_joint_pain",
    "neck hurts": "neck_pain",
    "neck ache": "neck_pain",
    "stiff neck": "stiff_neck",
    "eye pain": "pain_behind_the_eyes",
    "pain behind eyes": "pain_behind_the_eyes",
    "headache": "headache",
    "head hurts": "headache",
    "head pain": "headache",
    "migraine": "headache",
    "muscle ache": "muscle_pain",
    "body ache": "muscle_pain",
    "body pain": "muscle_pain",
    "joint ache": "joint_pain",
    "joint hurts": "joint_pain",
    "joints hurt": "joint_pain",
    "anal pain": "pain_in_anal_region",
    "rectal pain": "pain_in_anal_region",
    "pain when walking": "painful_walking",
    "hurts to walk": "painful_walking",
    "pain during bowel movement": "pain_during_bowel_movements",
    "painful pooping": "pain_during_bowel_movements",

    # DIGESTIVE
    "throwing up": "vomiting",
    "threw up": "vomiting",
    "puking": "vomiting",
    "nauseous": "nausea",
    "feel like vomiting": "nausea",
    "feel sick to stomach": "nausea",
    "queasy": "nausea",
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "loose stools": "diarrhoea",
    "runny stool": "diarrhoea",
    "watery stool": "diarrhoea",
    "constipated": "constipation",
    "can't poop": "constipation",
    "no bowel movement": "constipation",
    "gas": "passage_of_gases",
    "bloating": "distention_of_abdomen",
    "bloated stomach": "distention_of_abdomen",
    "swollen stomach": "swelling_of_stomach",
    "stomach swollen": "swelling_of_stomach",
    "heartburn": "acidity",
    "acid reflux": "acidity",
    "indigestion": "indigestion",
    "upset stomach": "indigestion",
    "no appetite": "loss_of_appetite",
    "not hungry": "loss_of_appetite",
    "lost appetite": "loss_of_appetite",
    "always hungry": "excessive_hunger",
    "very hungry": "excessive_hunger",
    "increased hunger": "excessive_hunger",
    "more appetite": "increased_appetite",
    "bloody stool": "bloody_stool",
    "blood in stool": "bloody_stool",
    "blood in poop": "bloody_stool",
    "stomach bleeding": "stomach_bleeding",

    # RESPIRATORY
    "coughing": "cough",
    "dry cough": "cough",
    "wet cough": "cough",
    "can't breathe": "breathlessness",
    "difficulty breathing": "breathlessness",
    "shortness of breath": "breathlessness",
    "out of breath": "breathlessness",
    "wheezing": "breathlessness",
    "runny nose": "runny_nose",
    "stuffy nose": "congestion",
    "nose congestion": "congestion",
    "blocked nose": "congestion",
    "sneezing": "continuous_sneezing",
    "cant stop sneezing": "continuous_sneezing",
    "sore throat": "throat_irritation",
    "throat hurts": "throat_irritation",
    "throat pain": "throat_irritation",
    "throat patches": "patches_in_throat",
    "mucus": "phlegm",
    "phlegm": "phlegm",
    "rusty mucus": "rusty_sputum",
    "blood in cough": "blood_in_sputum",
    "coughing blood": "blood_in_sputum",
    "sinus pain": "sinus_pressure",
    "sinus pressure": "sinus_pressure",

    # SKIN
    "itchy": "itching",
    "itching": "itching",
    "rash": "skin_rash",
    "skin rash": "skin_rash",
    "spots": "red_spots_over_body",
    "red spots": "red_spots_over_body",
    "blisters": "blister",
    "pimples": "pus_filled_pimples",
    "acne": "blackheads",
    "peeling skin": "skin_peeling",
    "skin peeling": "skin_peeling",
    "yellow skin": "yellowish_skin",
    "skin yellow": "yellowish_skin",
    "skin turning yellow": "yellowish_skin",
    "jaundiced": "yellowish_skin",
    "bruise": "bruising",
    "bruising": "bruising",
    "skin bumps": "nodal_skin_eruptions",
    "bumps on skin": "nodal_skin_eruptions",
    "dry lips": "drying_and_tingling_lips",
    "tingling lips": "drying_and_tingling_lips",
    "red nose sores": "red_sore_around_nose",
    "silver skin patches": "silver_like_dusting",
    "yellow crust on skin": "yellow_crust_ooze",
    "discolored patches": "dischromic _patches",
    "skin discoloration": "dischromic _patches",
    "nail pitting": "small_dents_in_nails",
    "brittle nails": "brittle_nails",
    "nail inflammation": "inflammatory_nails",

    # URINARY
    "peeing a lot": "polyuria",
    "urinating frequently": "polyuria",
    "frequent urination": "polyuria",
    "pee a lot": "polyuria",
    "burning when peeing": "burning_micturition",
    "burning urination": "burning_micturition",
    "pain when peeing": "burning_micturition",
    "bladder pain": "bladder_discomfort",
    "bladder pressure": "bladder_discomfort",
    "always feel like peeing": "continuous_feel_of_urine",
    "constant urge to pee": "continuous_feel_of_urine",
    "smelly urine": "foul_smell_of urine",
    "urine smells bad": "foul_smell_of urine",
    "dark urine": "dark_urine",
    "urine is dark": "dark_urine",
    "brown urine": "dark_urine",
    "yellow urine": "yellow_urine",
    "blood in urine": "spotting_ urination",
    "spotting when urinating": "spotting_ urination",

    # NEUROLOGICAL / MENTAL
    "dizzy": "dizziness",
    "dizziness": "dizziness",
    "lightheaded": "dizziness",
    "spinning": "spinning_movements",
    "room is spinning": "spinning_movements",
    "vertigo": "spinning_movements",
    "balance problems": "loss_of_balance",
    "losing balance": "loss_of_balance",
    "unsteady": "unsteadiness",
    "wobbly": "unsteadiness",
    "slurred speech": "slurred_speech",
    "speech problems": "slurred_speech",
    "can't concentrate": "lack_of_concentration",
    "hard to focus": "lack_of_concentration",
    "forgetful": "lack_of_concentration",
    "anxious": "anxiety",
    "anxiety": "anxiety",
    "nervous": "anxiety",
    "depressed": "depression",
    "depression": "depression",
    "mood swings": "mood_swings",
    "irritable": "irritability",
    "irritability": "irritability",
    "confused": "altered_sensorium",
    "confusion": "altered_sensorium",
    "disoriented": "altered_sensorium",
    "unconscious": "coma",
    "passed out": "coma",
    "visual problems": "visual_disturbances",
    "blurry vision": "blurred_and_distorted_vision",
    "vision blurry": "blurred_and_distorted_vision",
    "distorted vision": "blurred_and_distorted_vision",
    "loss of smell": "loss_of_smell",
    "can't smell": "loss_of_smell",
    "watery eyes": "watering_from_eyes",
    "eyes watering": "watering_from_eyes",
    "red eyes": "redness_of_eyes",
    "puffy eyes": "puffy_face_and_eyes",
    "puffy face": "puffy_face_and_eyes",
    "sunken eyes": "sunken_eyes",

    # ENERGY / GENERAL
    "tired": "fatigue",
    "exhausted": "fatigue",
    "fatigue": "fatigue",
    "weakness": "fatigue",
    "no energy": "fatigue",
    "lethargic": "lethargy",
    "lethargy": "lethargy",
    "sluggish": "lethargy",
    "malaise": "malaise",
    "generally unwell": "malaise",
    "feeling unwell": "malaise",
    "restless": "restlessness",
    "cant sleep": "restlessness",
    "dehydrated": "dehydration",
    "very thirsty": "dehydration",
    "weight loss": "weight_loss",
    "losing weight": "weight_loss",
    "weight gain": "weight_gain",
    "gaining weight": "weight_gain",
    "obese": "obesity",
    "overweight": "obesity",

    # CARDIOVASCULAR
    "heart racing": "fast_heart_rate",
    "racing heart": "fast_heart_rate",
    "heart pounding": "palpitations",
    "palpitations": "palpitations",
    "heart palpitations": "palpitations",
    "swollen legs": "swollen_legs",
    "legs swollen": "swollen_legs",
    "swollen ankles": "swollen_legs",
    "swollen hands": "swollen_extremeties",
    "swollen extremities": "swollen_extremeties",
    "varicose veins": "prominent_veins_on_calf",
    "veins on legs": "prominent_veins_on_calf",
    "fluid retention": "fluid_overload",

    # MUSCULOSKELETAL
    "stiff joints": "movement_stiffness",
    "stiffness": "movement_stiffness",
    "can't move joints": "movement_stiffness",
    "swollen joints": "swelling_joints",
    "muscle wasting": "muscle_wasting",
    "muscle loss": "muscle_wasting",
    "weak muscles": "muscle_weakness",
    "muscle weakness": "muscle_weakness",
    "weak arms": "weakness_in_limbs",
    "weak legs": "weakness_in_limbs",
    "limbs weak": "weakness_in_limbs",
    "one side weak": "weakness_of_one_body_side",
    "one side of body weak": "weakness_of_one_body_side",
    "paralysis": "weakness_of_one_body_side",
    "cramps": "cramps",
    "muscle cramps": "cramps",
    "leg cramps": "cramps",

    # THYROID / GLANDS
    "enlarged thyroid": "enlarged_thyroid",
    "swollen thyroid": "enlarged_thyroid",
    "goiter": "enlarged_thyroid",
    "swollen lymph nodes": "swelled_lymph_nodes",
    "swollen glands": "swelled_lymph_nodes",
    "lumps in neck": "swelled_lymph_nodes",

    # CHILLS / SWEATING
    "chills": "chills",
    "shivering": "shivering",
    "shaking": "shivering",
    "sweating": "sweating",
    "night sweats": "sweating",
    "excessive sweating": "sweating",
    "cold hands": "cold_hands_and_feets",
    "cold feet": "cold_hands_and_feets",
    "cold hands and feet": "cold_hands_and_feets",

    # MISC
    "blood transfusion": "receiving_blood_transfusion",
    "unsterile injection": "receiving_unsterile_injections",
    "drug use": "receiving_unsterile_injections",
    "family history": "family_history",
    "genetic history": "family_history",
    "blood sugar problems": "irregular_sugar_level",
    "sugar level issues": "irregular_sugar_level",
    "ulcers on tongue": "ulcers_on_tongue",
    "mouth ulcers": "ulcers_on_tongue",
    "tongue sores": "ulcers_on_tongue",
    "anal itching": "irritation_in_anus",
    "rectal itching": "irritation_in_anus",
    "internal itching": "internal_itching",
}


# ─────────────────────────────────────────────────────────────
# CORE MAPPER CLASS
# ─────────────────────────────────────────────────────────────
class SymptomMapper:
    """
    Maps natural language symptom descriptions to model vocabulary.

    Priority order:
    1. Exact match against vocabulary (e.g. "vomiting" → "vomiting")
    2. Synonym dictionary lookup (e.g. "throwing up" → "vomiting")
    3. Fuzzy match against vocabulary (e.g. "vomitin" → "vomiting")
    4. Fuzzy match against synonym keys (e.g. "threw up" → "vomiting")
    """

    def __init__(self, vocabulary: List[str] = None, threshold: int = 75):
        self.vocabulary = vocabulary or SYMPTOM_VOCABULARY
        self.threshold = threshold  # minimum fuzzy match score (0-100)

        # Build readable versions of vocab for fuzzy matching
        # "high_fever" → "high fever" (underscores → spaces)
        self.vocab_readable = {
            v.replace("_", " ").replace("  ", " ").strip(): v
            for v in self.vocabulary
        }
        # Synonym keys in lowercase for matching
        self.synonyms = {k.lower().strip(): v for k, v in SYNONYM_MAP.items()}

    def _normalize(self, text: str) -> str:
        """Lowercase, strip, remove filler phrases, normalize spaces."""
        text = text.lower().strip()
        # Remove common filler phrases
        fillers = [
            "i have", "i've been having", "i've had", "i am having", "i am",
            "i'm having", "i'm", "i feel", "i've", "experiencing", "suffering from",
            "complaining of", "having", "there is", "there's", "my", "a", "an",
            "some", "lot of", "lots of", "very", "really", "quite", "severe",
            "mild", "slight", "chronic", "acute",
        ]
        for filler in fillers:
            text = re.sub(r'\b' + re.escape(filler) + r'\b', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def map_symptom(self, raw_input: str) -> dict:
        """
        Map a single symptom string to model vocabulary.

        Returns:
            {
                "input": original input,
                "mapped": matched vocabulary term or None,
                "method": how it was matched,
                "score": confidence of match (100 = exact),
                "suggestions": list of close alternatives
            }
        """
        original = raw_input
        normalized = self._normalize(raw_input)

        # ── Step 1: Exact match against vocabulary ──
        if normalized.replace(" ", "_") in self.vocabulary:
            return {
                "input": original,
                "mapped": normalized.replace(" ", "_"),
                "method": "exact",
                "score": 100,
                "suggestions": []
            }

        # Also check with underscores as-is
        if normalized in self.vocabulary:
            return {
                "input": original,
                "mapped": normalized,
                "method": "exact",
                "score": 100,
                "suggestions": []
            }

        # ── Step 2: Synonym dictionary lookup ──
        if normalized in self.synonyms:
            return {
                "input": original,
                "mapped": self.synonyms[normalized],
                "method": "synonym",
                "score": 100,
                "suggestions": []
            }

        # ── Step 3: Fuzzy match against readable vocabulary ──
        vocab_keys = list(self.vocab_readable.keys())
        fuzzy_match = process.extractOne(
            normalized,
            vocab_keys,
            scorer=fuzz.token_sort_ratio
        )

        if fuzzy_match and fuzzy_match[1] >= self.threshold:
            matched_vocab = self.vocab_readable[fuzzy_match[0]]
            # Get top 3 alternatives for suggestions
            top3 = process.extract(normalized, vocab_keys, scorer=fuzz.token_sort_ratio, limit=3)
            suggestions = [self.vocab_readable[m[0]] for m in top3 if m[0] != fuzzy_match[0]]
            return {
                "input": original,
                "mapped": matched_vocab,
                "method": "fuzzy",
                "score": fuzzy_match[1],
                "suggestions": suggestions
            }

        # ── Step 4: Fuzzy match against synonym keys ──
        synonym_keys = list(self.synonyms.keys())
        syn_match = process.extractOne(
            normalized,
            synonym_keys,
            scorer=fuzz.token_sort_ratio
        )

        if syn_match and syn_match[1] >= self.threshold:
            mapped = self.synonyms[syn_match[0]]
            return {
                "input": original,
                "mapped": mapped,
                "method": "fuzzy_synonym",
                "score": syn_match[1],
                "suggestions": []
            }

        # ── Step 5: No match found — return suggestions anyway ──
        top_suggestions = process.extract(
            normalized, vocab_keys, scorer=fuzz.token_sort_ratio, limit=5
        )
        suggestions = [
            self.vocab_readable[m[0]] for m in top_suggestions
            if m[1] >= 40  # lower threshold just for suggestions
        ]

        return {
            "input": original,
            "mapped": None,
            "method": "none",
            "score": 0,
            "suggestions": suggestions
        }

    def map_symptoms(self, raw_inputs: List[str]) -> dict:
        """
        Map a list of symptom strings.

        Returns:
            {
                "mapped": list of vocabulary terms that matched,
                "results": detailed result for each input,
                "unmatched": inputs that couldn't be mapped,
                "warnings": list of warning messages
            }
        """
        mapped = []
        results = []
        unmatched = []
        warnings = []

        for raw in raw_inputs:
            result = self.map_symptom(raw)
            results.append(result)

            if result["mapped"]:
                # Avoid duplicates
                if result["mapped"] not in mapped:
                    mapped.append(result["mapped"])
                if result["method"] in ("fuzzy", "fuzzy_synonym"):
                    warnings.append(
                        f"'{raw}' was interpreted as '{result['mapped']}' "
                        f"(confidence: {result['score']}%). "
                        f"If incorrect, valid alternatives: {result['suggestions']}"
                    )
            else:
                unmatched.append(raw)
                if result["suggestions"]:
                    warnings.append(
                        f"'{raw}' was not recognized. "
                        f"Did you mean: {result['suggestions']}?"
                    )
                else:
                    warnings.append(
                        f"'{raw}' was not recognized and no close matches were found. "
                        f"Use GET /symptoms for the full list."
                    )

        return {
            "mapped": mapped,
            "results": results,
            "unmatched": unmatched,
            "warnings": warnings
        }


# ─────────────────────────────────────────────────────────────
# SINGLETON — import this in main.py
# ─────────────────────────────────────────────────────────────
mapper = SymptomMapper()