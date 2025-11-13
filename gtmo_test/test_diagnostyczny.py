#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("START TESTU")

try:
    import spacy
    print("1. spacy zaimportowane")
except Exception as e:
    print(f"BLAD importu spacy: {e}")
    exit(1)

try:
    nlp = spacy.load('pl_core_news_lg')
    print("2. Model pl_core_news_lg zaladowany")
except:
    try:
        nlp = spacy.load('pl_core_news_sm')
        print("2. Model pl_core_news_sm zaladowany")
    except Exception as e:
        print(f"BLAD ladowania modelu: {e}")
        exit(1)

try:
    doc = nlp("Test zdania.")
    print(f"3. Analiza dziala, tokenow: {len(doc)}")
    
    for token in doc:
        print(f"   Token: {token.text}, POS: {token.pos_}, DEP: {token.dep_}")
        
except Exception as e:
    print(f"BLAD analizy: {e}")

print("KONIEC TESTU")