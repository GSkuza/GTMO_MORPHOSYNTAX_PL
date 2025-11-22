# ⚡ Quick Start Guide

## 🚀 Uruchomienie lokalne (2 minuty)

### Windows

```bash
# 1. Ustaw klucz API
set ANTHROPIC_API_KEY=your_api_key_here

# 2. Uruchom
demo_webapp\local_test.bat
```

### Linux/Mac

```bash
# 1. Ustaw klucz API
export ANTHROPIC_API_KEY=your_api_key_here

# 2. Uruchom
chmod +x demo_webapp/local_test.sh
./demo_webapp/local_test.sh
```

### Otwórz w przeglądarce

- Frontend: http://localhost:8080
- API Docs: http://localhost:8000/docs

---

## 🌐 Deploy na Railway (5 minut)

```bash
# 1. Zainstaluj Railway CLI
npm install -g railway

# 2. Zaloguj się
railway login

# 3. Utwórz projekt
railway init

# 4. Ustaw klucz API
railway variables set ANTHROPIC_API_KEY=your_key

# 5. Deploy
railway up

# 6. Skopiuj URL backendu
railway status
# Przykład: https://gtmo-production.railway.app
```

### Konfiguruj frontend

Edytuj `demo_webapp/docs/js/main.js`:

```javascript
const API_BASE_URL = 'https://gtmo-production.railway.app';
```

### Włącz GitHub Pages

1. Push do GitHub: `git push origin main`
2. Settings → Pages → Source: `/demo_webapp/docs`
3. Gotowe! Strona pod: `https://username.github.io/GTMO_MORPHOSYNTAX/`

---

## 📊 Użycie

1. **Wgraj dokument** - Kliknij "Wybierz plik..." (obsługuje .txt i .md)
2. **Analizuj** - Kliknij "Analizuj dokument" (trwa 2-5 min)
3. **Zobacz wyniki**:
   - 📊 Statystyki dokumentu
   - 📈 Tabela metryk konstytucyjnych
   - 💡 Rekomendacje od Claude

---

## 🔧 Testowanie API

```bash
# Test health
curl http://localhost:8000/health

# Test analysis
python demo_webapp/test_api.py http://localhost:8000 demo_webapp/docs/sample_document.txt
```

---

## 📚 Dokumentacja

- [README.md](README.md) - Pełna dokumentacja
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Szczegółowy przewodnik wdrożenia
- [API Docs](http://localhost:8000/docs) - Interaktywna dokumentacja API (Swagger)

---

## ❓ Problemy?

### Backend nie startuje

```bash
# Sprawdź zależności
pip install -r demo_webapp/api/requirements.txt

# Sprawdź logi
python demo_webapp/api/main.py
```

### CORS error

- Upewnij się że URL backendu w `main.js` jest poprawny
- Sprawdź czy backend działa: `curl http://localhost:8000/health`

### Brak rekomendacji

- Sprawdź czy `ANTHROPIC_API_KEY` jest ustawiony: `echo $ANTHROPIC_API_KEY`
- Sprawdź czy zaznaczyłeś "Użyj LLM" w formularzu

---

## 🎯 Następne kroki

Po uruchomieniu możesz:

1. ✅ Przetestować z własnymi dokumentami
2. 🎨 Dostosować wygląd (edytuj `docs/css/styles.css`)
3. 🚀 Wdrożyć na Railway/Render/Heroku
4. 📊 Dodać więcej wizualizacji
5. 🔐 Dodać autentykację dla produkcji

---

**Miłej zabawy z GTMØ! 🎉**
