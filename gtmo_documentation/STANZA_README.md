# GTMØ + Stanza Integration

## 🚀 Quick Start (5 minut)

### 1. Instalacja
```bash
pip install stanza
python -c "import stanza; stanza.download('pl')"
```

### 2. Użycie
```python
from gtmo_morphosyntax import EnhancedGTMOProcessor

processor = EnhancedGTMOProcessor()
text = "Sąd uznał, że oskarżony nie popełnił czynu. Jednak go skazuje."

result = processor.analyze_legal_text(text)

# Sprawdź smoking guns (sprzeczności)
if result['stanza_analysis']['smoking_guns']:
    for gun in result['stanza_analysis']['smoking_guns']:
        print(f"🔫 {gun['type']}: {gun['details']['conflict']}")
        print(f"   Severity: {gun['severity']:.2f}")

# Ocena jakości
print(f"\nQuality: {result['legal_assessment']['quality']}")
print(f"Coherence: {result['legal_assessment']['legal_coherence_score']:.2f}")
```

### 3. Output (JSON)
```json
{
  "gtmo_coordinates": {
    "determination": 0.87,
    "stability": 0.23,
    "entropy": 0.91
  },
  "stanza_analysis": {
    "smoking_guns": [
      {
        "type": "negation_conflict",
        "severity": 0.98,
        "details": {
          "conflict": "'nie popełnił' → 'skazuje'"
        }
      }
    ]
  },
  "legal_assessment": {
    "quality": "critical",
    "legal_coherence_score": 0.12
  }
}
```

---

## 🎯 Co To Daje?

### 1. Smoking Guns - Automatyczne Wykrywanie Sprzeczności
**Przed:** Prawnik musi czytać 2-3 razy, żeby zauważyć "nie popełnił" → "skazuje"
**Po:** System wykrywa w 1 sekundę z severity 0.98

### 2. Causal Chain Validation
**Wykrywa:** Złamane łańcuchy logiczne, rozumowanie kołowe

### 3. Temporal Consistency
**Wykrywa:** Paradoksy czasowe (np. "zatrzymanie przed przestępstwem")

### 4. GTMØ Coordinates
**Mapuje** sprzeczności na współrzędne [D, S, E] i sprawdza odległość od singularności (paradoksu)

---

## 📊 Format JSON

Pełna specyfikacja: [`enhanced_json_schema.json`](enhanced_json_schema.json)
Przykład: [`example_stanza_output.json`](example_stanza_output.json)

Najważniejsze pola:
- `stanza_analysis.smoking_guns[]` - **wykryte sprzeczności**
- `causality_analysis.causal_strength` - siła argumentacji (0-1)
- `temporal_analysis.paradoxes[]` - paradoksy czasowe
- `legal_assessment.quality` - ocena jakości (excellent/good/fair/poor/critical)
- `legal_assessment.legal_coherence_score` - spójność (0-1)
- `singularity_warning.active` - czy zbliżenie do paradoksu

---

## 🔧 Migracja Starych JSON

Masz stare pliki w formacie GTMØ 3.0? Użyj:

```bash
# Pojedynczy plik z pełną reanalizą
python migrate_to_stanza_format.py old_result.json --reanalyze

# Cały folder
python migrate_to_stanza_format.py --dir ./results --reanalyze
```

---

## 🐛 Troubleshooting

**Problem:** "Stanza not available"
```bash
pip install stanza
python -c "import stanza; stanza.download('pl')"
```

**Problem:** Wolna analiza
- Użyj GPU: `stanza.Pipeline('pl', use_gpu=True)`
- Dziel długie teksty na chunki po ~5000 znaków

**Problem:** Out of memory
- Przetwarzaj małe partie (10-20 dokumentów)
- Zwolnij pamięć między partiami: `del results`

---

## 📚 Więcej Informacji

- **Demo:** `python gtmo_morphosyntax.py`
- **JSON Schema:** [`enhanced_json_schema.json`](enhanced_json_schema.json)
- **Przykład:** [`example_stanza_output.json`](example_stanza_output.json)
- **Migracja:** [`migrate_to_stanza_format.py`](migrate_to_stanza_format.py)
