# GTMØ Constitutional Metrics Analyzer - Web Demo

Demo aplikacji webowej do analizy morfosyntaktycznej dokumentów prawnych z wykorzystaniem metryk konstytucyjnych GTMØ.

## 🌟 Funkcjonalności

1. **Wgrywanie dokumentów** - Obsługa plików .txt i .md z treścią ustaw
2. **Analiza morfosyntaktyczna** - Automatyczna analiza dokumentu przy użyciu GTMØ Quantum Morphosyntax Engine
3. **Tabela metryk** - Przejrzysta prezentacja metryk konstytucyjnych (SA, D-S-E, CI, itp.)
4. **Rekomendacje LLM** - Wyjaśnienie wyników w języku naturalnym z wykorzystaniem Claude API

## 📁 Struktura projektu

```
demo_webapp/
├── api/                    # Backend (FastAPI)
│   ├── main.py            # Główny plik API
│   └── requirements.txt   # Zależności backendu
├── docs/                  # Frontend (GitHub Pages)
│   ├── index.html        # Strona główna
│   ├── css/
│   │   └── styles.css    # Style CSS
│   └── js/
│       └── main.js       # Logika frontendu
└── README.md             # Ten plik
```

## 🚀 Deployment

### Backend (Railway / Render / Heroku)

1. **Railway (Zalecane)**

   ```bash
   # 1. Zainstaluj Railway CLI
   npm install -g railway

   # 2. Zaloguj się
   railway login

   # 3. Utwórz nowy projekt
   railway init

   # 4. Dodaj zmienne środowiskowe
   railway variables set ANTHROPIC_API_KEY=your_api_key_here

   # 5. Deploy
   railway up
   ```

2. **Render**

   - Utwórz konto na [render.com](https://render.com)
   - Kliknij "New +" → "Web Service"
   - Podłącz repozytorium GitHub
   - Ustaw:
     - Build Command: `pip install -r demo_webapp/api/requirements.txt`
     - Start Command: `cd demo_webapp/api && uvicorn main:app --host 0.0.0.0 --port $PORT`
     - Environment Variables: `ANTHROPIC_API_KEY=your_key`

3. **Heroku**

   ```bash
   # 1. Utwórz Procfile w głównym katalogu
   echo "web: cd demo_webapp/api && uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile

   # 2. Deploy
   heroku create gtmo-analyzer
   heroku config:set ANTHROPIC_API_KEY=your_api_key_here
   git push heroku main
   ```

### Frontend (GitHub Pages)

1. **Zaktualizuj URL API**

   Edytuj `docs/js/main.js` i zmień `API_BASE_URL`:

   ```javascript
   const API_BASE_URL = 'https://your-deployed-backend.railway.app';
   ```

2. **Włącz GitHub Pages**

   - Przejdź do Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main` / Folder: `/demo_webapp/docs`
   - Save

3. **Adres strony**

   Twoja strona będzie dostępna pod:
   `https://your-username.github.io/GTMO_MORPHOSYNTAX/`

## 🔧 Lokalne uruchomienie

### Backend

```bash
# 1. Przejdź do katalogu API
cd demo_webapp/api

# 2. Zainstaluj zależności
pip install -r requirements.txt

# 3. Ustaw klucz API
export ANTHROPIC_API_KEY=your_api_key_here  # Linux/Mac
# lub
set ANTHROPIC_API_KEY=your_api_key_here     # Windows

# 4. Uruchom serwer
python main.py
```

Backend będzie dostępny pod `http://localhost:8000`

### Frontend

```bash
# Przejdź do katalogu frontendu
cd demo_webapp/docs

# Uruchom prosty serwer HTTP
python -m http.server 8080
```

Frontend będzie dostępny pod `http://localhost:8080`

## 📊 Przykład użycia

1. Otwórz stronę w przeglądarce
2. Kliknij "Wybierz plik..." i wgraj dokument (.txt lub .md)
3. (Opcjonalnie) Odznacz "Użyj LLM" jeśli chcesz szybszą analizę bez rekomendacji
4. Kliknij "Analizuj Dokument"
5. Czekaj na wyniki (2-5 minut w zależności od długości dokumentu)

## 🔑 Wymagany klucz API

Aplikacja wymaga klucza API Anthropic (Claude) do generowania rekomendacji.

Uzyskaj klucz na: [console.anthropic.com](https://console.anthropic.com)

## ⚙️ Konfiguracja

### Zmienne środowiskowe

- `ANTHROPIC_API_KEY` - Klucz API Anthropic (wymagany dla rekomendacji LLM)

### Limity

- Maksymalny rozmiar pliku: ~10 MB
- Timeout analizy: 5 minut
- Format plików: .txt, .md

## 🐛 Rozwiązywanie problemów

### Błąd CORS

Jeśli widzisz błędy CORS, upewnij się że:
- Backend ma poprawnie skonfigurowany CORS middleware
- URL backendu w `main.js` jest poprawny

### Timeout podczas analizy

Dla bardzo długich dokumentów (>50 stron):
- Zwiększ timeout w `api/main.py` (domyślnie 300s)
- Rozważ podzielenie dokumentu na mniejsze części

### Brak rekomendacji

Jeśli nie widzisz rekomendacji:
- Sprawdź czy `ANTHROPIC_API_KEY` jest ustawiony
- Sprawdź logi backendu: `railway logs` lub `heroku logs --tail`

## 📄 Licencja

GTMØ Quantum Morphosyntax Engine © 2025

## 🔗 Linki

- [Dokumentacja GTMØ](../README.md)
- [GitHub Repository](https://github.com/yourusername/GTMO_MORPHOSYNTAX)
- [Anthropic API](https://www.anthropic.com)
