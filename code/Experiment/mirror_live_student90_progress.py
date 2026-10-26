"""Keep the canonical Student progress Markdown current across run handoffs."""
import json
import time
from pathlib import Path

ROOT = Path('/root/Diffusion 3')
EXT = ROOT / 'FID/student90_guidance32_extension_20261022/status.json'
MAIN = ROOT / 'FID/student90_cfg25_restart17/status.json'
SOURCE = ROOT / 'docs/STUDENT90_INTERNALIZATION_EXTENSION.md'
TARGET = ROOT / 'docs/STUDENT90_PROGRESS.md'


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def sync():
    if not SOURCE.exists():
        return
    data = SOURCE.read_text(encoding='utf-8')
    if not TARGET.exists() or TARGET.read_text(encoding='utf-8') != data:
        temp = TARGET.with_suffix('.tmp')
        temp.write_text(data, encoding='utf-8')
        temp.replace(TARGET)


def main():
    while True:
        ext, main_run = read(EXT), read(MAIN)
        sync()
        if ext.get('status') == 'completed' and main_run.get('status') == 'running':
            # The main trainer owns STUDENT90_PROGRESS.md again after restart.
            time.sleep(15)
            latest = read(MAIN)
            if latest.get('status') == 'running' and latest.get('updated_utc') != main_run.get('updated_utc'):
                return
        time.sleep(5)


if __name__ == '__main__':
    main()
