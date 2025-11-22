# 🚀 Przewodnik Wdrożenia GTMØ Web Demo

## Szybki start (Railway + GitHub Pages)

### Krok 1: Przygotowanie repozytorium

```bash
# Jeśli jeszcze nie masz repo na GitHubie
cd GTMO_MORPHOSYNTAX
git add demo_webapp/
git commit -m "Add GTMØ web demo"
git push origin main
```

### Krok 2: Deploy backendu na Railway

1. **Utwórz konto na Railway**
   - Przejdź na [railway.app](https://railway.app)
   - Zaloguj się przez GitHub

2. **Utwórz nowy projekt**
   - Kliknij "New Project"
   - Wybierz "Deploy from GitHub repo"
   - Wybierz repozytorium `GTMO_MORPHOSYNTAX`

3. **Konfiguracja**
   - Railway automatycznie wykryje `railway.json`
   - Dodaj zmienne środowiskowe:
     - Kliknij "Variables"
     - Dodaj `ANTHROPIC_API_KEY` = `twoj_klucz_api`

4. **Deploy**
   - Railway automatycznie zbuduje i wdroży aplikację
   - Po zakończeniu, skopiuj URL (np. `https://gtmo-analyzer-production.railway.app`)

### Krok 3: Konfiguracja frontendu

1. **Zaktualizuj URL API w frontencie**

   Edytuj `demo_webapp/docs/js/main.js`:

   ```javascript
   const API_BASE_URL = window.location.hostname === 'localhost'
       ? 'http://localhost:8000'
       : 'https://gtmo-analyzer-production.railway.app'; // ← Twój URL z Railway
   ```

2. **Commit i push**

   ```bash
   git add demo_webapp/docs/js/main.js
   git commit -m "Update API URL"
   git push origin main
   ```

### Krok 4: Włącz GitHub Pages

1. **Ustawienia repozytorium**
   - Przejdź do Settings → Pages
   - Source: "Deploy from a branch"
   - Branch: `main`
   - Folder: `/demo_webapp/docs` ⚠️ **Ważne!**
   - Kliknij "Save"

2. **Czekaj na deployment**
   - GitHub zbuduje stronę (1-2 minuty)
   - URL: `https://twoj-username.github.io/GTMO_MORPHOSYNTAX/`

3. **Gotowe!** 🎉
   - Otwórz URL w przeglądarce
   - Wgraj przykładowy dokument i przetestuj

---

## Alternatywne opcje deploymentu

### Backend: Render.com

1. **Utwórz konto na Render**
   - [render.com](https://render.com)

2. **Nowy Web Service**
   - "New +" → "Web Service"
   - Podłącz GitHub repo

3. **Konfiguracja**
   - Build Command: `pip install -r demo_webapp/api/requirements.txt`
   - Start Command: `cd demo_webapp/api && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Environment Variables:
     - `ANTHROPIC_API_KEY` = `twoj_klucz`

4. **Deploy**
   - Render zbuduje i wdroży aplikację

### Backend: Heroku

```bash
# 1. Zaloguj się do Heroku
heroku login

# 2. Utwórz aplikację
heroku create gtmo-analyzer

# 3. Dodaj buildpack Python
heroku buildpacks:set heroku/python

# 4. Ustaw zmienne środowiskowe
heroku config:set ANTHROPIC_API_KEY=twoj_klucz

# 5. Deploy
git push heroku main

# 6. Otwórz aplikację
heroku open
```

### Backend: PythonAnywhere

1. **Utwórz konto**
   - [pythonanywhere.com](https://www.pythonanywhere.com)

2. **Upload kodu**
   - Użyj "Files" → "Upload a file"
   - Lub sklonuj repo przez Bash console

3. **Zainstaluj zależności**
   ```bash
   pip install --user -r demo_webapp/api/requirements.txt
   ```

4. **Konfiguruj Web App**
   - Web → "Add a new web app"
   - Framework: Manual configuration
   - Python version: 3.10
   - WSGI file: skonfiguruj dla FastAPI/uvicorn

---

## Weryfikacja deploymentu

### Test backendu

```bash
# Health check
curl https://twoj-backend-url.railway.app/health

# Test API (wymaga pliku testowego)
python demo_webapp/test_api.py https://twoj-backend-url.railway.app
```

### Test frontendu

1. Otwórz `https://twoj-username.github.io/GTMO_MORPHOSYNTAX/`
2. Wgraj plik `demo_webapp/docs/sample_document.txt`
3. Kliknij "Analizuj dokument"
4. Sprawdź czy widzisz wyniki

---

## Rozwiązywanie problemów

### ❌ CORS Error

**Problem:** Błąd CORS w konsoli przeglądarki

**Rozwiązanie:**
1. Sprawdź czy backend ma poprawnie skonfigurowany CORS w `api/main.py`
2. Upewnij się że URL w `docs/js/main.js` jest poprawny (bez końcowego `/`)
3. Sprawdź czy backend działa: `curl https://backend-url/health`

### ❌ 404 Not Found na GitHub Pages

**Problem:** Strona nie ładuje się

**Rozwiązanie:**
1. Sprawdź czy wybrałeś właściwy folder: `/demo_webapp/docs`
2. Sprawdź czy `index.html` jest w głównym katalogu `docs/`
3. Czekaj 2-3 minuty na propagację

### ❌ Timeout podczas analizy

**Problem:** Analiza przerywa się po 30 sekundach

**Rozwiązanie:**
1. Na Railway: Zwiększ timeout w `railway.json` (healthcheckTimeout)
2. W kodzie: Zwiększ timeout w `api/main.py` (subprocess timeout)
3. Rozważ podzielenie dużych dokumentów

### ❌ Brak rekomendacji

**Problem:** Tabela się ładuje, ale brak rekomendacji

**Rozwiązanie:**
1. Sprawdź czy `ANTHROPIC_API_KEY` jest ustawiony na backendzie
2. Sprawdź logi backendu: `railway logs` lub `heroku logs --tail`
3. Sprawdź czy zaznaczono "Użyj LLM" w formularzu

---

## Monitorowanie

### Railway
```bash
# Logi
railway logs

# Status
railway status
```

### Heroku
```bash
# Logi
heroku logs --tail

# Status
heroku ps
```

### Render
- Dashboard → Logs (w interfejsie webowym)

---

## Koszt deploymentu

| Platforma | Darmowy plan | Limit |
|-----------|-------------|-------|
| Railway | $5 credit/miesiąc | ~500 godzin uruchomienia |
| Render | 750h/miesiąc | Sleeps po 15 min nieaktywności |
| Heroku | 550-1000h/miesiąc | Sleeps po 30 min nieaktywności |
| GitHub Pages | Unlimited | Tylko statyczne pliki |

**Rekomendacja:** Railway dla początku (najłatwiejszy setup)

---

## Bezpieczeństwo

### ⚠️ Ważne zasady:

1. **NIE commituj kluczy API do repozytorium!**
   - Używaj zmiennych środowiskowych
   - Dodaj `.env` do `.gitignore`

2. **Ogranicz dostęp do API**
   - Rozważ dodanie rate limiting
   - Dodaj authentication dla produkcji

3. **Monitoruj koszty API**
   - Anthropic API jest płatny
   - Ustaw limity w Anthropic Console

---

## Następne kroki

Po wdrożeniu możesz:

1. **Dodać własne style** - Edytuj `docs/css/styles.css`
2. **Rozszerzyć API** - Dodaj więcej endpointów w `api/main.py`
3. **Cache wyników** - Dodaj Redis dla cachowania analiz
4. **Autentykacja** - Dodaj OAuth2 dla zabezpieczenia
5. **Database** - Zapisuj wyniki analiz do PostgreSQL

---

## Pomoc

- 📧 Issues: [GitHub Issues](https://github.com/yourusername/GTMO_MORPHOSYNTAX/issues)
- 📚 Dokumentacja: [README.md](README.md)
- 🤝 Community: [Discussions](https://github.com/yourusername/GTMO_MORPHOSYNTAX/discussions)
