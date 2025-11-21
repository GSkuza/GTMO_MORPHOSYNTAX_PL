#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('gtmo_results/analysis_21112025_no1_NEW_ustawa_o_trzezwosci/full_document.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Wyciągnij wszystkie zdania
all_sentences = []
for article in data['articles']:
    if 'sentences' in article:
        for sent in article['sentences']:
            all_sentences.append(sent)

# Próbka z "Grupy #5" (to są numery ZDAŃ 1-indexed w raporcie, więc -1 dla 0-indexed)
sample_nums = [0, 4, 10, 18, 121, 126, 138, 146]  # 0-indexed

print('Zdanie# | Glob.# | Tekst')
print('-' * 100)

for num in sample_nums:
    if num < len(all_sentences):
        sent = all_sentences[num]
        text = sent.get('content', {}).get('text', '')[:80]
        glob_num = sent.get('analysis_metadata', {}).get('sentence_number', '?')
        print(f'{num+1:7d} | {glob_num:6} | {text}')

print(f'\nCałkowita liczba zdań: {len(all_sentences)}')
