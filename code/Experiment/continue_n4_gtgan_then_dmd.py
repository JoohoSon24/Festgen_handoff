"""Continue N4 GT-GAN for 1K updates from its optimizer resume, then unlock N1 DMD."""
import fcntl
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path('/root/Diffusion 3')
EXP = ROOT / 'Experiment'
BASE = Path('/root/Diffusion/code/workspace_distillation')
OUT = ROOT / 'FID/student90_n1_budget_extension_20261025'
N1DMD = ROOT / 'FID/student90_n1_dmd_after_n4_20261026'
CONFIG_PATH = EXP / 'configs/exp-20261021-student90-adaptive.json'
MAIN_DOC = ROOT / 'docs/STUDENT90_PROGRESS.md'
DOC = OUT / 'progress.md'
sys.path[:0] = [str(EXP), str(EXP / 'source'), str(BASE)]

import torch
import student90_adaptive as trainer
from teacher.diffusion import Diffusion
import adaptive_reproducibility as repro


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def report_wrapper(original):
    def report_and_mirror():
        original()
        temp = MAIN_DOC.with_suffix('.tmp')
        temp.write_text(DOC.read_text(encoding='utf-8'), encoding='utf-8')
        temp.replace(MAIN_DOC)
    return report_and_mirror


def main():
    assert torch.cuda.is_available(), 'CUDA is required for the N4 continuation.'
    config = json.loads(CONFIG_PATH.read_text())
    trainer.C = config
    trainer.CONFIG_PATH = CONFIG_PATH
    trainer.OUT = OUT
    trainer.DOC = DOC
    trainer.POLICY = config['adaptive_policy']
    trainer.REVISION = repro.capture_revision(ROOT, OUT, CONFIG_PATH, Path(__file__))
    trainer.S = {'status': 'running', 'pid': __import__('os').getpid(),
                 'phase': 'n4_gtgan_continuation', 'current_stage': 'gan_n4_extension', 'stages': {}}
    trainer.report = report_wrapper(trainer.report)
    trainer.alpha = Diffusion(1000, .0001, .02, torch.device('cuda')).alpha_bar
    trainer.GAN_TARGET_STEPS = 2000
    trainer.GAN_EVAL_UPDATES = {1250, 1500, 1750, 2000}

    prior = json.loads((OUT / 'gan_n4/result.json').read_text())
    source = prior['source']
    resume = OUT / 'gan_n4/best_resume.pt'
    extension = OUT / 'gan_n4_extension'
    extension.mkdir(parents=True, exist_ok=True)
    latest = extension / 'latest.pt'
    if not latest.exists():
        shutil.copy2(resume, latest)
    saved = torch.load(latest, map_location='cpu', weights_only=False)
    assert saved['update'] == 1000 and saved['warm'] == 100
    assert saved['source'] == source and saved['best']['update'] == 1000
    assert saved['evaluations'][-1]['update'] == 1000
    resume_best_metrics = saved['best']['metrics']
    del saved
    baseline = torch.load(source, map_location='cpu', weights_only=False)
    trainer.student_config = baseline['student_config']
    del baseline

    trainer.S['stages']['n1'] = {'status': 'completed', 'update': 112500, 'budget': 112500,
        'best_metrics': json.loads((OUT / 'n1_result.json').read_text())['best']['metrics'],
        'evaluations': json.loads((OUT / 'n1_result.json').read_text())['evaluations']}
    trainer.S['stages']['gan_n4'] = {'status': 'completed', 'update': 1000, 'budget': 1000,
        'best_metrics': prior['best']['metrics'], 'evaluations': prior['evaluations']}
    trainer.S['stages']['gan_n4_extension'] = {'status': 'resuming', 'update': 1000,
        'budget': 2000, 'best_metrics': resume_best_metrics,
        'evaluations': prior['evaluations']}

    with (EXP / 'gt_gan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = trainer.gan_finetune('n4_extension', source)
        write(OUT / 'n4_extension_completed.json', {
            'status': 'completed', 'prior_stage_result': str(OUT / 'gan_n4/result.json'),
            'resume_checkpoint': str(resume), 'resume_sha256': repro.shared.sha(resume),
            'added_updates': 1000, 'total_updates_from_original_anchor': result['best']['update'],
            'result': result, 'n4_best_checkpoint': result['best']['checkpoint'],
            'created_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        })
        (extension / 'latest.pt').unlink(missing_ok=True)
        trainer.S.update({'status': 'completed', 'phase': 'n4_gtgan_continuation_completed',
                          'current_stage': None})
        trainer.S['stages']['gan_n4_extension'].update({'status': 'completed',
            'update': result.get('target_updates', 2000), 'budget': result.get('target_updates', 2000),
            'best_metrics': result['best']['metrics'], 'evaluations': result['evaluations']})
        trainer.report()
    # The queue will acquire this same lock, then run DMD using the newly selected N4 endpoint.


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / 'n4_extension_failure.json', {'status': 'failed',
              'traceback': __import__('traceback').format_exc(),
              'updated_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())})
        raise
