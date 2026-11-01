import re
import shutil
from pathlib import Path

target = Path(__file__).resolve().parent / "backend" / "app" / "db" / "session.py"
text = target.read_text(encoding="utf-8")

if 'or "clip"' in text:
    print("Already patched - nothing changed.")
else:
    pattern = re.compile(r"^([ \t]*)architecture_family[ \t]*=[ \t]*([^\n]*?),[ \t]*$", re.MULTILINE)
    new_text, n = pattern.subn(r'\1architecture_family=(\2) or "clip",', text)
    if n == 0:
        print("No 'architecture_family=' line found in session.py - nothing changed.")
    else:
        shutil.copy(target, target.with_name("session.py.bak2"))
        target.write_text(new_text, encoding="utf-8")
        print(f"Fixed: patched {n} 'architecture_family' line(s) in {target}")