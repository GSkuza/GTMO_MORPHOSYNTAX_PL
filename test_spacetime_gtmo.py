# -*- coding: utf-8 -*-
"""
Test czasoprzestrzeni semantycznej GTMØ i kompozycji morfemów
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from geometric_dse_computer import (
    GeometricDSEComputer, DSEResult,
    compute_semantic_interval, classify_semantic_relation, IntervalType,
    compose_morphemes, analyze_morpheme_derivation,
    POLISH_AFFIX_DELTAS, KAPPA_SEMANTIC, classify_to_attractor
)

print('=' * 80)
print('TEST: Czasoprzestrzen GTMO i kompozycja morfemow')
print('=' * 80)

# =============================================================================
# TEST 1: INTERWAŁ SEMANTYCZNY (Minkowski)
# =============================================================================
print('\n' + '=' * 80)
print('1. INTERWAL SEMANTYCZNY (ds^2 = -kappa^2*dS^2 + dD^2 + dE^2)')
print('=' * 80)
print(f'   KAPPA_SEMANTIC = {KAPPA_SEMANTIC}')

# Przypadek SPACELIKE: duża różnica w D i E, mała w S
p1_space = (0.8, 0.5, 0.2)  # wysoka D, średnie S, niskie E
p2_space = (0.3, 0.5, 0.8)  # niska D, średnie S, wysokie E
interval_space = compute_semantic_interval(p1_space, p2_space)
print(f'\n   SPACELIKE (roznica kontekstowa):')
print(f'   P1 = {p1_space} -> P2 = {p2_space}')
print(f'   ds^2 = -{KAPPA_SEMANTIC}^2*({interval_space.dS})^2 + ({interval_space.dD})^2 + ({interval_space.dE})^2')
print(f'   ds^2 = {interval_space.ds_squared:.6f}')
print(f'   Typ: {interval_space.interval_type.value}')

# Przypadek TIMELIKE: duża różnica w S, mała w D i E
p1_time = (0.5, 0.9, 0.3)   # średnie D, wysokie S, niskie E
p2_time = (0.5, 0.2, 0.3)   # średnie D, niskie S, niskie E
interval_time = compute_semantic_interval(p1_time, p2_time)
print(f'\n   TIMELIKE (ewolucja temporalna):')
print(f'   P1 = {p1_time} -> P2 = {p2_time}')
print(f'   ds^2 = -{KAPPA_SEMANTIC}^2*({interval_time.dS})^2 + ({interval_time.dD})^2 + ({interval_time.dE})^2')
print(f'   ds^2 = {interval_time.ds_squared:.6f}')
print(f'   Typ: {interval_time.interval_type.value}')

# Przypadek LIGHTLIKE: |dS| ≈ sqrt(dD^2 + dE^2)
p1_light = (0.5, 0.5, 0.5)
p2_light = (0.8, 1.0, 0.9)  # dD=0.3, dS=0.5, dE=0.4
interval_light = compute_semantic_interval(p1_light, p2_light)
print(f'\n   LIGHTLIKE (przejscie fazowe):')
print(f'   P1 = {p1_light} -> P2 = {p2_light}')
print(f'   ds^2 = {interval_light.ds_squared:.6f}')
print(f'   Typ: {interval_light.interval_type.value}')

# =============================================================================
# TEST 2: KOMPOZYCJA MORFEMÓW
# =============================================================================
print('\n' + '=' * 80)
print('2. KOMPOZYCJA MORFEMOW: Phi(slowo) = Phi(rdzen) + Sum(Delta(afiks))')
print('=' * 80)

# Rdzeń: 'pis-' (pisać) - stabilny, zdeterminowany czasownik
root_pis = (0.7, 0.8, 0.3)
print(f'\n   Rdzen: "pis-" (pisac)')
print(f'   Phi(pis) = [D={root_pis[0]}, S={root_pis[1]}, E={root_pis[2]}]')

# Derywacja: pis- + -anie -> pisanie
comp1 = compose_morphemes(root_pis, ['-anie'], root_name='pis')
print(f'\n   Derywacja 1: pis- + -anie -> pisanie')
print(f'   Delta(-anie) = {POLISH_AFFIX_DELTAS["-anie"]}')
print(f'   Phi(pisanie) = {comp1.composed_dse}')

# Derywacja: nie- + pis- + -anie -> niepisanie
comp2 = compose_morphemes(root_pis, ['-anie', 'nie-'], root_name='pis')
print(f'\n   Derywacja 2: nie- + pis- + -anie -> niepisanie')
print(f'   Delta(-anie) = {POLISH_AFFIX_DELTAS["-anie"]}')
print(f'   Delta(nie-) = {POLISH_AFFIX_DELTAS["nie-"]}')
print(f'   Phi(niepisanie) = {comp2.composed_dse}')

# Derywacja: pis- + -nik -> pisnik
comp3 = compose_morphemes(root_pis, ['-nik'], root_name='pis')
print(f'\n   Derywacja 3: pis- + -nik -> pisnik')
print(f'   Delta(-nik) = {POLISH_AFFIX_DELTAS["-nik"]}')
print(f'   Phi(pisnik) = {comp3.composed_dse}')

# =============================================================================
# TEST 3: WPŁYW NA D-S-E - ANALIZA PORÓWNAWCZA
# =============================================================================
print('\n' + '=' * 80)
print('3. WPLYW KOMPOZYCJI NA D-S-E')
print('=' * 80)

print('\n   Porownanie: rdzen vs derywaty')
print('   ' + '-' * 60)
print(f'   {"Morfem":<20} {"D":>8} {"S":>8} {"E":>8} {"Atraktor":>20}')
print('   ' + '-' * 60)

# Rdzeń
dse_root = DSEResult(D=root_pis[0], S=root_pis[1], E=root_pis[2])
attr_root, _ = classify_to_attractor(dse_root)
print(f'   {"pis- (rdzen)":<20} {root_pis[0]:>8.4f} {root_pis[1]:>8.4f} {root_pis[2]:>8.4f} {attr_root:>20}')

# pisanie
dse_pisanie = comp1.to_DSEResult()
attr_pisanie, _ = classify_to_attractor(dse_pisanie)
print(f'   {"pisanie":<20} {dse_pisanie.D:>8.4f} {dse_pisanie.S:>8.4f} {dse_pisanie.E:>8.4f} {attr_pisanie:>20}')

# niepisanie
dse_niepisanie = comp2.to_DSEResult()
attr_niepisanie, _ = classify_to_attractor(dse_niepisanie)
print(f'   {"niepisanie":<20} {dse_niepisanie.D:>8.4f} {dse_niepisanie.S:>8.4f} {dse_niepisanie.E:>8.4f} {attr_niepisanie:>20}')

# pisnik
dse_pisnik = comp3.to_DSEResult()
attr_pisnik, _ = classify_to_attractor(dse_pisnik)
print(f'   {"pisnik":<20} {dse_pisnik.D:>8.4f} {dse_pisnik.S:>8.4f} {dse_pisnik.E:>8.4f} {attr_pisnik:>20}')

# =============================================================================
# TEST 4: INTERWAŁY MIĘDZY DERYWATAMI
# =============================================================================
print('\n' + '=' * 80)
print('4. INTERWALY SEMANTYCZNE MIEDZY DERYWATAMI')
print('=' * 80)

pairs = [
    ('pis-', root_pis, 'pisanie', comp1.composed_dse),
    ('pisanie', comp1.composed_dse, 'niepisanie', comp2.composed_dse),
    ('pis-', root_pis, 'pisnik', comp3.composed_dse),
]

for name1, p1, name2, p2 in pairs:
    interval = compute_semantic_interval(p1, p2)
    print(f'\n   {name1} <-> {name2}:')
    print(f'   ds^2 = {interval.ds_squared:>10.6f}')
    print(f'   Typ:  {interval.interval_type.value:>10}')
    print(f'   dD={interval.dD:+.4f}, dS={interval.dS:+.4f}, dE={interval.dE:+.4f}')

# =============================================================================
# TEST 5: ANALIZA DERYWACJI
# =============================================================================
print('\n' + '=' * 80)
print('5. ANALIZA DERYWACJI MORFOLOGICZNEJ')
print('=' * 80)

analysis = analyze_morpheme_derivation(
    root_pis, comp2.composed_dse,
    base_name='pis-', derived_name='niepisanie'
)
print(f'\n   Derywacja: {analysis["base_name"]} -> {analysis["derived_name"]}')
print(f'   Estymowana delta afiksu: {analysis["estimated_delta"]}')
print(f'   Typ transformacji: {analysis["transformation_type"]}')
print(f'   Wplyw na D-S-E:')
print(f'     D: {analysis["impact"]["D"]}')
print(f'     S: {analysis["impact"]["S"]}')
print(f'     E: {analysis["impact"]["E"]}')

# =============================================================================
# TEST 6: PRZYKŁAD Z PRAWDZIWYMI EMBEDDINGAMI
# =============================================================================
print('\n' + '=' * 80)
print('6. TEST Z SYMULOWANYMI EMBEDDINGAMI')
print('=' * 80)

np.random.seed(42)

# Symulacja embeddingu dla 'woda' (jednoznaczny)
centroid_woda = np.random.randn(768) * 0.5
emb_woda = centroid_woda + np.random.randn(30, 768) * 0.1

# Symulacja embeddingu dla 'wodny' (derywat)
emb_wodny = centroid_woda + np.random.randn(30, 768) * 0.12 + np.random.randn(768) * 0.05

computer_woda = GeometricDSEComputer(emb_woda)
computer_wodny = GeometricDSEComputer(emb_wodny)

dse_woda = computer_woda.compute_DSE()
dse_wodny = computer_wodny.compute_DSE()

print(f'\n   "woda" (rdzen): D={dse_woda.D:.4f}, S={dse_woda.S:.4f}, E={dse_woda.E:.4f}')
print(f'   "wodny" (derywat): D={dse_wodny.D:.4f}, S={dse_wodny.S:.4f}, E={dse_wodny.E:.4f}')

interval_real = compute_semantic_interval(dse_woda, dse_wodny)
print(f'\n   Interwal semantyczny:')
print(f'   ds^2 = {interval_real.ds_squared:.6f}')
print(f'   Typ: {interval_real.interval_type.value}')

# Porównanie z teoretyczną deltą dla -ny
delta_ny = POLISH_AFFIX_DELTAS['-ny']
predicted_wodny = np.array([dse_woda.D, dse_woda.S, dse_woda.E]) + delta_ny
print(f'\n   Przewidywany "wodny" z delta(-ny): [D={predicted_wodny[0]:.4f}, S={predicted_wodny[1]:.4f}, E={predicted_wodny[2]:.4f}]')
print(f'   Rzeczywisty "wodny":                [D={dse_wodny.D:.4f}, S={dse_wodny.S:.4f}, E={dse_wodny.E:.4f}]')

error = np.linalg.norm(predicted_wodny - np.array([dse_wodny.D, dse_wodny.S, dse_wodny.E]))
print(f'   Blad predykcji: {error:.4f}')

# =============================================================================
# PODSUMOWANIE WPŁYWU NA D-S-E
# =============================================================================
print('\n' + '=' * 80)
print('PODSUMOWANIE: WPLYW CZASOPRZESTRZENI NA D-S-E')
print('=' * 80)

print('''
KLUCZOWE WNIOSKI:

1. INTERWAŁ SEMANTYCZNY nie zmienia wartości D-S-E, ale dodaje nową METRYKĘ
   do porównywania morfemów w przestrzeni F³.

2. KLASYFIKACJA INTERWAŁÓW:
   - SPACELIKE (I² > 0): Gdy różnice w D i E dominują nad S
     -> Dwa niezależne znaczenia (np. "zamek" jako budowla vs "zamek" błyskawiczny)

   - TIMELIKE (I² < 0): Gdy różnica w S dominuje
     -> Ewolucja znaczenia (np. "fajny" dawniej vs "fajny" dziś)

   - LIGHTLIKE (I² ≈ 0): Przejście fazowe
     -> Granica semantyczna (np. metafora stająca się dosłownym znaczeniem)

3. KOMPOZYCJA MORFEMÓW wpływa na D-S-E przez ADDYTYWNE DELTY:
   - Prefiksy negacyjne (nie-): obniżają D, podnoszą E
   - Sufiksy nominalizacyjne (-anie, -enie): lekko obniżają D, podnoszą E
   - Sufiksy agentywne (-nik, -arz): podnoszą D i S

4. STABILNOŚĆ (S) JAKO WYMIAR CZASOWY:
   - W metryce Minkowskiego S ma znak UJEMNY
   - Duże zmiany S = ewolucja temporalna (timelike)
   - To odzwierciedla, że S mierzy "trwałość w czasie"
''')

print('=' * 80)
print('WSZYSTKIE TESTY CZASOPRZESTRZENI GTMO ZAKONCZONE!')
print('=' * 80)
