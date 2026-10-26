"""Resume the planned N4 GT-GAN follow-up without repeating completed N1."""
import fcntl
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path('/root/Diffusion 3')
EXP = ROOT / 'Experiment'
BASE = ROOT / 'Diffusion/code/workspace_distillation'
OUT = ROOT / 'FID/student90_n1_budget_extension_20261025'
SOURCE_RUN = ROOT / 'FID/student90_n8_n4_n2_budget_extension_20261023'
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


def summary(path):
    result = json.loads(path.read_text())
    logs = path.parent / 'train.jsonl'
    last = json.loads(logs.read_text().splitlines()[-1]) if logs.exists() and logs.stat().st_size else {}
    return {'status': 'completed', 'update': result['training_updates'], 'budget': result['base_updates'],
            'evaluations': result['evaluations'], 'best_metrics': result['best']['metrics'], 'last_log': last}


def main():
    assert torch.cuda.is_available(), 'CUDA is required for N4 GT-GAN fine-tuning.'
    config = json.loads(CONFIG_PATH.read_text())
    trainer.C = config
    trainer.CONFIG_PATH = CONFIG_PATH
    trainer.OUT = OUT
    trainer.DOC = DOC
    trainer.POLICY = config['adaptive_policy']
    trainer.S = {'status': 'running', 'pid': __import__('os').getpid(),
                 'phase': 'gt_gan_fine_tuning', 'current_stage': 'gan_n4', 'stages': {}}
    trainer.S['stages']['n1'] = summary(OUT / 'n1_result.json')
    for name in ['n8', 'n4', 'n2']:
        result_path = SOURCE_RUN / name / 'result.json'
        trainer.S['stages'][name] = summary(result_path)
    original_report = trainer.report

    def report_and_mirror():
        original_report()
        temp = MAIN_DOC.with_suffix('.tmp')
        temp.write_text(DOC.read_text(encoding='utf-8'), encoding='utf-8')
        temp.replace(MAIN_DOC)

    trainer.report = report_and_mirror
    trainer.REVISION = repro.capture_revision(ROOT, OUT, CONFIG_PATH, Path(__file__))
    trainer.alpha = Diffusion(1000, .0001, .02, torch.device('cuda')).alpha_bar
    n4_result = json.loads((SOURCE_RUN / 'n4/result.json').read_text())
    trainer.student_config = torch.load(n4_result['selected_checkpoint'], map_location='cpu', weights_only=False)['student_config']
    folder = OUT / 'n4'
    folder.mkdir(parents=True, exist_ok=True)
    lock_path = EXP / 'gt_gan.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        n4_baseline = trainer.export_and_verify(n4_result)
        write(OUT / 'n4_baseline_source.json', {'source_run': str(SOURCE_RUN / 'n4/result.json'),
              'selected_checkpoint': n4_result['selected_checkpoint'], 'selected_update': n4_result['best']['update'],
              'official_fid': n4_result['best']['metrics']['official_fid'], 'exported_checkpoint': n4_baseline})
        result = trainer.gan_finetune('n4', n4_baseline)
        gan_folder = OUT / 'gan_n4'
        for feature_path in gan_folder.glob('eval_*/features.npy'):
            feature_path.unlink(missing_ok=True)
        (gan_folder / 'latest.pt').unlink(missing_ok=True)
        trainer.S.update({'status': 'completed', 'phase': 'completed', 'current_stage': None})
        write(OUT / 'n4_gtgan_completed.json', {'n4_baseline_checkpoint': n4_baseline, 'result': result})
        report_and_mirror()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        trainer.S.update({'status': 'failed', 'error': __import__('traceback').format_exc()})
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / 'n4_gtgan_failure.json', trainer.S)
        try:
            trainer.report()
        except Exception:
            pass
        raise
