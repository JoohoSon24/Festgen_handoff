"""Pause at a durable checkpoint, extend guidance32, then resume the main run."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/Diffusion 3')
RUN = ROOT / 'FID/student90_cfg25_restart17'
OUT = ROOT / 'FID/student90_guidance32_extension_20261022'
PYTHON = Path('/root/Diffusion/code/workspace_distillation/.venv/bin/python')
MAIN = ROOT / 'Experiment/student90_adaptive.py'
EXTENSION = ROOT / 'Experiment/continue_guidance32_internalization.py'
LOG = ROOT / 'Experiment/student90_internalization_extension.log'


def read(path):
    return json.loads(path.read_text())


def proc_alive(pid):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().split()[2]
        return state != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    initial_status = read(RUN / 'status.json')
    pid = int(initial_status['pid'])
    resume_path = Path(read(RUN / 'active_resume.json')['path'])
    initial_mtime = resume_path.stat().st_mtime_ns
    print(f"Waiting for a newer durable checkpoint from pid={pid}; current stage={initial_status['current_stage']}.", flush=True)
    while proc_alive(pid):
        status = read(RUN / 'status.json')
        active = read(RUN / 'active_resume.json')
        candidate = Path(active['path'])
        if candidate.exists() and candidate.stat().st_mtime_ns > initial_mtime:
            print(f"Durable checkpoint recorded at update={active.get('update')}; pausing original run.", flush=True)
            os.kill(pid, signal.SIGTERM)
            break
        time.sleep(5)
    deadline = time.time() + 180
    while proc_alive(pid) and time.time() < deadline:
        time.sleep(2)
    if proc_alive(pid):
        raise RuntimeError(f'Original training process {pid} did not stop after SIGTERM; no second GPU job launched.')
    active = read(RUN / 'active_resume.json')
    record = {
        'original_pid': pid, 'paused_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'durable_checkpoint': active['path'], 'checkpoint_update': active.get('update'),
        'resume_after_extension': 'restart the original driver from its saved stage checkpoint',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'main_run_pause.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    with LOG.open('a', encoding='utf-8') as log:
        log.write('\n=== CFG internalization extension ===\n')
        log.flush()
        result = subprocess.run([str(PYTHON), '-u', str(EXTENSION)], cwd=str(ROOT / 'Experiment'), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f'Guidance extension exited with code {result.returncode}; original run remains resumable.')
    with LOG.open('a', encoding='utf-8') as log:
        log.write('\n=== Resume original NFE pipeline ===\n')
        log.flush()
        resumed = subprocess.run([str(PYTHON), '-u', str(MAIN)], cwd=str(ROOT / 'Experiment'), stdout=log, stderr=subprocess.STDOUT)
    if resumed.returncode != 0:
        raise RuntimeError(f'Original NFE pipeline resume exited with code {resumed.returncode}.')


if __name__ == '__main__':
    main()
