#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMO Truth Observer (τ) - LLM-as-Judge Component
=================================================

Komponent weryfikacji prawdziwości twierdzeń (τ) używający LLM jako sędziego.
Działa jako drugi etap pipeline'u GTMO:

    GTMO (strukturalny) → τ (weryfikacja faktyczna)

Typy twierdzeń:
- FACTUAL: Weryfikowalne fakty (τ ∈ [0,1])
- OPINION: Subiektywne opinie (τ = N/A)
- WORDPLAY: Gry słowne/kalambury (τ = 0.0, special case)
- NONSENSE: Semantyczny nonsens (τ = 0.0)
- HARMFUL: Szkodliwe/rasistowskie (τ = 0.0, flagged)

Użycie:
    from gtmo_truth_observer import LLMTruthObserver

    observer = LLMTruthObserver(api_key="...")
    result = observer.verify("Ziemia krąży wokół Słońca")
    # {'truth_value': 0.95, 'confidence': 0.9, 'claim_type': 'FACTUAL', ...}
"""

import os
import json
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class ClaimType(Enum):
    """Typy twierdzeń rozpoznawane przez obserwator τ."""
    FACTUAL = "FACTUAL"       # Weryfikowalny fakt
    OPINION = "OPINION"       # Subiektywna opinia
    WORDPLAY = "WORDPLAY"     # Gra słowna/kalambur
    NONSENSE = "NONSENSE"     # Semantyczny nonsens
    HARMFUL = "HARMFUL"       # Szkodliwe/rasistowskie
    UNKNOWN = "UNKNOWN"       # Nie można określić


@dataclass
class TruthVerdict:
    """Wynik weryfikacji prawdziwości."""
    truth_value: Optional[float]  # τ ∈ [0,1] lub None dla opinii
    confidence: float             # Pewność oceny [0,1]
    claim_type: ClaimType         # Typ twierdzenia
    reasoning: str                # Uzasadnienie
    is_verifiable: bool           # Czy twierdzenie jest weryfikowalne
    flags: Dict[str, bool]        # Dodatkowe flagi (harmful, misleading, etc.)

    def to_dict(self) -> dict:
        """Konwertuje do słownika dla JSON."""
        return {
            'truth_value': self.truth_value,
            'confidence': self.confidence,
            'claim_type': self.claim_type.value,
            'reasoning': self.reasoning,
            'is_verifiable': self.is_verifiable,
            'flags': self.flags
        }


class LLMTruthObserver:
    """
    Obserwator prawdziwości τ używający LLM jako sędziego.

    Wspiera:
    - Anthropic Claude API (preferowany)
    - OpenAI API (fallback)
    - Lokalny mock (dla testów bez API)
    """

    SYSTEM_PROMPT = """Jesteś ekspertem od weryfikacji faktów i analizy twierdzeń w języku polskim.

Twoim zadaniem jest ocena PRAWDZIWOŚCI twierdzenia według następującej skali:

TYPY TWIERDZEŃ:
- FACTUAL: Weryfikowalny fakt (np. "Ziemia krąży wokół Słońca")
- OPINION: Subiektywna opinia (np. "Pizza jest smaczna")
- WORDPLAY: Gra słowna/kalambur (np. "Włoszka to mieszkanka włoszczyzny")
- NONSENSE: Semantyczny nonsens (np. "Stolica Warszawy to Polska")
- HARMFUL: Szkodliwe/rasistowskie/dyskryminujące (np. twierdzenia o wyższości ras)

SKALA PRAWDZIWOŚCI (τ):
- τ = 1.0: W pełni prawdziwe, zweryfikowany fakt
- τ = 0.8-0.9: Prawdziwe z drobnymi nieścisłościami
- τ = 0.5-0.7: Częściowo prawdziwe, wymaga kontekstu
- τ = 0.2-0.4: Głównie fałszywe, zawiera elementy prawdy
- τ = 0.0-0.1: Całkowicie fałszywe lub nonsens
- τ = null: Nie dotyczy (opinie, pytania)

ODPOWIEDŹ w formacie JSON:
{
  "claim_type": "FACTUAL|OPINION|WORDPLAY|NONSENSE|HARMFUL",
  "truth_value": 0.0-1.0 lub null,
  "confidence": 0.0-1.0,
  "reasoning": "Krótkie uzasadnienie po polsku",
  "is_verifiable": true/false,
  "flags": {
    "harmful": false,
    "misleading": false,
    "requires_context": false
  }
}"""

    USER_PROMPT_TEMPLATE = """Oceń prawdziwość następującego twierdzenia:

"{text}"

Odpowiedz TYLKO w formacie JSON, bez dodatkowego tekstu."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "auto",  # "anthropic", "openai", "mock"
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500
    ):
        """
        Inicjalizuje obserwator τ.

        Args:
            api_key: Klucz API (lub z env: ANTHROPIC_API_KEY / OPENAI_API_KEY)
            provider: Dostawca LLM ("anthropic", "openai", "mock", "auto")
            model: Model do użycia (domyślnie: claude-3-haiku / gpt-4o-mini)
            temperature: Temperatura generacji (niższa = bardziej deterministyczny)
            max_tokens: Maksymalna liczba tokenów odpowiedzi
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.model = model
        self.api_key = api_key

        # Auto-detect provider
        if provider == "auto":
            self._auto_detect_provider()
        else:
            self._setup_provider(provider, api_key, model)

    def _auto_detect_provider(self):
        """Automatycznie wykrywa dostępnego dostawcę API."""
        # Try Anthropic first
        anthropic_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic
                self.provider = "anthropic"
                self.api_key = anthropic_key
                self.model = self.model or "claude-3-haiku-20240307"
                self.client = anthropic.Anthropic(api_key=anthropic_key)
                return
            except ImportError:
                pass

        # Try OpenAI
        openai_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                self.provider = "openai"
                self.api_key = openai_key
                self.model = self.model or "gpt-4o-mini"
                self.client = openai.OpenAI(api_key=openai_key)
                return
            except ImportError:
                pass

        # Fallback to mock
        print("⚠️  No API key found. Using mock provider (rule-based heuristics).")
        self.provider = "mock"
        self.client = None

    def _setup_provider(self, provider: str, api_key: Optional[str], model: Optional[str]):
        """Konfiguruje wybranego dostawcę."""
        if provider == "anthropic":
            import anthropic
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.model = model or "claude-3-haiku-20240307"
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif provider == "openai":
            import openai
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.model = model or "gpt-4o-mini"
            self.client = openai.OpenAI(api_key=self.api_key)
        elif provider == "mock":
            self.client = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def verify(self, text: str) -> TruthVerdict:
        """
        Weryfikuje prawdziwość twierdzenia.

        Args:
            text: Tekst do weryfikacji

        Returns:
            TruthVerdict z wynikami weryfikacji
        """
        if self.provider == "mock":
            return self._mock_verify(text)
        elif self.provider == "anthropic":
            return self._anthropic_verify(text)
        elif self.provider == "openai":
            return self._openai_verify(text)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _anthropic_verify(self, text: str) -> TruthVerdict:
        """Weryfikacja przez Anthropic Claude API."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(text=text)}
                ]
            )

            response_text = message.content[0].text
            return self._parse_response(response_text)

        except Exception as e:
            print(f"⚠️  Anthropic API error: {e}")
            return self._fallback_verdict(text, str(e))

    def _openai_verify(self, text: str) -> TruthVerdict:
        """Weryfikacja przez OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(text=text)}
                ]
            )

            response_text = response.choices[0].message.content
            return self._parse_response(response_text)

        except Exception as e:
            print(f"⚠️  OpenAI API error: {e}")
            return self._fallback_verdict(text, str(e))

    def _parse_response(self, response_text: str) -> TruthVerdict:
        """Parsuje odpowiedź JSON z LLM."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group())

            # Parse claim type
            claim_type_str = data.get('claim_type', 'UNKNOWN').upper()
            try:
                claim_type = ClaimType(claim_type_str)
            except ValueError:
                claim_type = ClaimType.UNKNOWN

            # Parse truth value
            truth_value = data.get('truth_value')
            if truth_value is not None:
                truth_value = float(truth_value)
                truth_value = max(0.0, min(1.0, truth_value))  # Clamp to [0,1]

            return TruthVerdict(
                truth_value=truth_value,
                confidence=float(data.get('confidence', 0.5)),
                claim_type=claim_type,
                reasoning=data.get('reasoning', ''),
                is_verifiable=data.get('is_verifiable', True),
                flags=data.get('flags', {})
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️  Parse error: {e}")
            return self._fallback_verdict(response_text, str(e))

    def _fallback_verdict(self, text: str, error: str) -> TruthVerdict:
        """Zwraca domyślny werdykt w przypadku błędu."""
        return TruthVerdict(
            truth_value=None,
            confidence=0.0,
            claim_type=ClaimType.UNKNOWN,
            reasoning=f"Error during verification: {error}",
            is_verifiable=False,
            flags={'error': True}
        )

    def _mock_verify(self, text: str) -> TruthVerdict:
        """
        Mock weryfikacja oparta na heurystykach (dla testów bez API).

        Rozpoznaje typowe wzorce:
        - Gry słowne (Włoszka, włoszczyzna, etc.)
        - Nonsens semantyczny (stolica X to Y)
        - Twierdzenia rasistowskie
        - Podstawowe fakty naukowe
        """
        text_lower = text.lower()

        # === HARMFUL CONTENT ===
        harmful_patterns = [
            r'ras[ay]?\s+(jest|są)\s+(mądrzejsz|lepsz|gorsz|głupsz)',
            r'(biał[ay]|czarn[ay]|żółt[ay])\s+ras[ay]',
            r'(żydzi|murzyni|cyganie)\s+(są|to)\s+',
            r'wyższość\s+(ras|narodow)',
        ]
        for pattern in harmful_patterns:
            if re.search(pattern, text_lower):
                return TruthVerdict(
                    truth_value=0.0,
                    confidence=0.95,
                    claim_type=ClaimType.HARMFUL,
                    reasoning="Twierdzenie zawiera treści rasistowskie/dyskryminujące",
                    is_verifiable=False,
                    flags={'harmful': True, 'racist': True}
                )

        # === WORDPLAY / KALAMBUR ===
        wordplay_patterns = [
            (r'włoszk[ai].*włoszczyzn', "Kalambur: Włoszka/włoszczyzna"),
            (r'niemk[ai].*niem', "Kalambur: Niemka/Niemcy"),
            (r'francuzk[ai].*franc', "Kalambur: Francuzka/Francja"),
            (r'mieszkank[ai]', "Potencjalny kalambur z narodowością"),
        ]
        for pattern, reason in wordplay_patterns:
            if re.search(pattern, text_lower):
                return TruthVerdict(
                    truth_value=0.0,
                    confidence=0.85,
                    claim_type=ClaimType.WORDPLAY,
                    reasoning=reason,
                    is_verifiable=False,
                    flags={'wordplay': True}
                )

        # === NONSENSE ===
        # "Stolica X to Y" gdzie X to miasto (odwrócona relacja)
        nonsense_match = re.search(r'stolic[aąę]\s+(\w+)\s+(to|jest)\s+', text_lower)
        if nonsense_match:
            # Check if it's reversed (city as country)
            cities = ['warszawy', 'krakowa', 'gdańska', 'poznania', 'wrocławia']
            if any(city in text_lower for city in cities):
                if not re.search(r'stolic[aąę]\s+(polski|niemiec|francji)', text_lower):
                    return TruthVerdict(
                        truth_value=0.0,
                        confidence=0.9,
                        claim_type=ClaimType.NONSENSE,
                        reasoning="Odwrócona relacja stolica-kraj (nonsens semantyczny)",
                        is_verifiable=True,
                        flags={'semantic_error': True}
                    )

        # === VERIFIABLE FACTS ===
        # Scientific facts
        scientific_facts = [
            (r'ziemia\s+krąży\s+wokół\s+słońca', 1.0, "Fakt astronomiczny"),
            (r'woda\s+wrze\s+w\s+100', 0.95, "Fakt fizyczny (zależy od ciśnienia)"),
            (r'2\s*\+\s*2\s*=\s*4', 1.0, "Fakt matematyczny"),
            (r'polska\s+.*\s+unii\s+europejskiej', 0.95, "Fakt geopolityczny"),
            (r'konstytucj[aię]\s+.*\s+1997', 0.95, "Fakt prawny"),
        ]
        for pattern, truth, reason in scientific_facts:
            if re.search(pattern, text_lower):
                return TruthVerdict(
                    truth_value=truth,
                    confidence=0.85,
                    claim_type=ClaimType.FACTUAL,
                    reasoning=reason,
                    is_verifiable=True,
                    flags={}
                )

        # Sports facts (harder to verify without knowledge)
        if re.search(r'(piłk[aię]|zawodnik|drużyn|mecz|gol)', text_lower):
            return TruthVerdict(
                truth_value=0.7,
                confidence=0.5,
                claim_type=ClaimType.FACTUAL,
                reasoning="Twierdzenie sportowe - wymaga weryfikacji",
                is_verifiable=True,
                flags={'requires_context': True}
            )

        # === OPINIONS ===
        opinion_markers = [
            r'\b(myślę|uważam|sądzę|moim\s+zdaniem)\b',
            r'\b(dobr[yae]|zł[yae]|piękn[yae]|brzydki[ae]|smaczn[yae])\b',
            r'\b(powinien|powinno|należy)\b',
            r'\b(lepiej|gorzej|najlepszy|najgorszy)\b',
        ]
        for pattern in opinion_markers:
            if re.search(pattern, text_lower):
                return TruthVerdict(
                    truth_value=None,
                    confidence=0.7,
                    claim_type=ClaimType.OPINION,
                    reasoning="Twierdzenie zawiera subiektywną ocenę",
                    is_verifiable=False,
                    flags={}
                )

        # === DEFAULT: UNKNOWN ===
        return TruthVerdict(
            truth_value=0.5,
            confidence=0.3,
            claim_type=ClaimType.UNKNOWN,
            reasoning="Nie można jednoznacznie sklasyfikować (mock provider)",
            is_verifiable=True,
            flags={'uncertain': True}
        )

    def batch_verify(self, texts: list) -> list:
        """
        Weryfikuje listę tekstów.

        Args:
            texts: Lista tekstów do weryfikacji

        Returns:
            Lista TruthVerdict
        """
        return [self.verify(text) for text in texts]


def combine_gtmo_and_truth(
    gtmo_result: dict,
    truth_verdict: TruthVerdict,
    weights: Tuple[float, float] = (0.6, 0.4)
) -> dict:
    """
    Łączy wyniki GTMO z weryfikacją prawdziwości.

    Args:
        gtmo_result: Wynik analizy GTMO (z coordinates, adelic, etc.)
        truth_verdict: Wynik weryfikacji τ
        weights: Wagi (gtmo_weight, truth_weight) dla combined score

    Returns:
        Rozszerzony wynik z sekcją 'truth' i 'combined_verdict'
    """
    result = gtmo_result.copy()

    # Add truth section
    result['truth'] = truth_verdict.to_dict()

    # Combined verdict logic
    adelic_emerged = gtmo_result.get('adelic', {}).get('emerged', False)
    adelic_status = gtmo_result.get('adelic', {}).get('status', 'unknown')
    truth_value = truth_verdict.truth_value
    claim_type = truth_verdict.claim_type

    # Determine combined verdict
    if claim_type == ClaimType.HARMFUL:
        combined_verdict = "REJECTED_HARMFUL"
        combined_confidence = 0.95
    elif claim_type == ClaimType.NONSENSE:
        combined_verdict = "REJECTED_NONSENSE"
        combined_confidence = 0.9
    elif claim_type == ClaimType.WORDPLAY:
        combined_verdict = "WORDPLAY_DETECTED"
        combined_confidence = 0.85
    elif claim_type == ClaimType.OPINION:
        combined_verdict = "OPINION_NO_TRUTH_VALUE"
        combined_confidence = truth_verdict.confidence
    elif adelic_emerged and truth_value is not None and truth_value >= 0.7:
        combined_verdict = "VALIDATED"
        combined_confidence = min(truth_verdict.confidence, 0.9)
    elif adelic_emerged and truth_value is not None and truth_value < 0.3:
        combined_verdict = "STRUCTURALLY_COHERENT_BUT_FALSE"
        combined_confidence = truth_verdict.confidence
    elif not adelic_emerged and truth_value is not None and truth_value >= 0.7:
        combined_verdict = "TRUE_BUT_SEMANTICALLY_UNSTABLE"
        combined_confidence = min(truth_verdict.confidence, 0.7)
    elif adelic_status == 'borderline':
        combined_verdict = "BORDERLINE"
        combined_confidence = 0.5
    else:
        combined_verdict = "UNCERTAIN"
        combined_confidence = 0.3

    result['combined_verdict'] = {
        'verdict': combined_verdict,
        'confidence': combined_confidence,
        'gtmo_emerged': adelic_emerged,
        'gtmo_status': adelic_status,
        'truth_value': truth_value,
        'claim_type': claim_type.value
    }

    return result


# === CLI Interface ===

def main():
    """CLI do testowania obserwatora τ."""
    import sys
    import io

    # Windows encoding fix
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("  GTMO Truth Observer (τ) - LLM-as-Judge")
    print("=" * 70)

    # Test texts
    test_texts = [
        "Ziemia krąży wokół Słońca.",
        "Włoszka to mieszkanka włoszczyzny.",
        "Stolica Warszawy to od 1678 roku Polska.",
        "Biała rasa jest mądrzejsza od czarnej.",
        "Pizza jest smaczna.",
        "W Polsce obowiązuje Konstytucja RP z 1997 roku.",
        "Drużyna piłki nożnej składa się z 11 zawodników.",
        "2 + 2 = 5",
    ]

    # Initialize observer
    observer = LLMTruthObserver(provider="mock")  # Use mock for demo
    print(f"\n Provider: {observer.provider}")
    print(f" Model: {observer.model or 'mock heuristics'}\n")

    for text in test_texts:
        print(f"\n📝 \"{text}\"")
        verdict = observer.verify(text)

        # Display result
        if verdict.truth_value is not None:
            tau_display = f"τ = {verdict.truth_value:.2f}"
        else:
            tau_display = "τ = N/A"

        type_icons = {
            ClaimType.FACTUAL: "📊",
            ClaimType.OPINION: "💭",
            ClaimType.WORDPLAY: "🎭",
            ClaimType.NONSENSE: "❓",
            ClaimType.HARMFUL: "🚫",
            ClaimType.UNKNOWN: "❔",
        }
        icon = type_icons.get(verdict.claim_type, "❔")

        print(f"   {icon} Type: {verdict.claim_type.value}")
        print(f"   {tau_display} (confidence: {verdict.confidence:.2f})")
        print(f"   Reasoning: {verdict.reasoning}")

        if verdict.flags:
            flags_str = ", ".join(f"{k}={v}" for k, v in verdict.flags.items() if v)
            if flags_str:
                print(f"   Flags: {flags_str}")

    print("\n" + "=" * 70)
    print(" Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
