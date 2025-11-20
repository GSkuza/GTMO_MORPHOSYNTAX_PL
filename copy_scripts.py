import os
import shutil
import glob

# Utwórz folder docelowy
target_dir = r"D:\GTMO_MORPHOSYNTAX\gtmo_python_scripts"
os.makedirs(target_dir, exist_ok=True)

# Znajdź wszystkie pliki .py w głównym katalogu
source_dir = r"D:\GTMO_MORPHOSYNTAX"
py_files = glob.glob(os.path.join(source_dir, "*.py"))

# Kopiuj pliki
copied = 0
for file in py_files:
    if os.path.basename(file) != "copy_scripts.py":  # Pomijamy ten tymczasowy skrypt
        shutil.copy2(file, target_dir)
        copied += 1
        print(f"Skopiowano: {os.path.basename(file)}")

print(f"\nŁącznie skopiowano {copied} plików .py do {target_dir}")
