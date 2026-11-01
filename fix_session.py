import re
import shutil
from pathlib import Path

target = Path(__file__).resolve().parent / "backend" / "app" / "db" / "session.py"
text = target.read_text(encoding="utf-8")

# Remove the invalid "is_default=..." keyword argument line(s)
new_text, n = re.subn(r"^[ \t]*is_default[ \t]*=[^\n]*\n", "", text, flags=re.MULTILINE)

if n == 0:
    print("No 'is_default=' line found in session.py - nothing changed.")
else:
    shutil.copy(target, target.with_suffix(".py.bak"))
    target.write_text(new_text, encoding="utf-8")
    print(f"Fixed: removed {n} 'is_default' line(s) from {target}")
    print("Backup saved as session.py.bak")