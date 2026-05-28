from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SCRIPT = ROOT / 'tools' / 'build_index_ent.py'


if __name__ == '__main__':
  sys.path.insert(0, str(ROOT))
  runpy.run_path(str(CANONICAL_SCRIPT), run_name='__main__')
