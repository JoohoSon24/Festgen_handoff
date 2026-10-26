"""Extend N8, then re-distill N4 and N2 from each parent's best Official-FID EMA."""
import copy
import fcntl
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path('/root/Diffusion 3')
BASE = Path('/root/Diffusion/code/workspace_distillation')
EXP = ROOT / 'Experiment'
SOURCE = ROOT / 'FID/student90_cfg25_restart17'
OUT = ROOT / 'FID/student90_n8_n4_n2_budget_extension_20261023'
CONFIG_PATH = EXP / 'configs/exp-20261021-student90-adaptive.json'
MAIN_DOC = ROOT / 'docs/STUDENT90_PROGRESS.md'
DOC = ROOT / 'docs/STUDENT90_BUDGET_EXTENSION.md'
PYTHON = Path('/root/Diffusion/code/workspace_distillation/.venv/bin/python')

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


def mirror_report():
    original_report()
    temp = MAIN_DOC.with_suffix('.tmp')
    temp.write_text(trainer.DOC.read_text(encoding='utf-8'), encoding='utf-8')
    temp.replace(MAIN_DOC)


def branch_room():
    free = shutil.disk_usage(ROOT).free
    floor = int(1.5 * 1024**3)
    if free < floor:
        raise RuntimeError(f'Disk free {free / 1024**3:.2f} GiB < extension safety floor {floor / 1024**3:.1f} GiB.')


def prepare_n8_resume(parent):
    source_folder = SOURCE / 'n8'
    stage_folder = OUT / 'n8'
    stage_folder.mkdir(parents=True, exist_ok=True)
    latest = stage_folder / 'latest.pt'
    selected = stage_folder / 'selected.pt'
    if not latest.exists():
        source_resume = source_folder / 'best_resume.pt'
        state = torch.load(source_resume, map_location='cpu', weights_only=False)
        assert state['stage'] == 'n8' and int(state['update']) == 20000
        assert state['parent'] == str(parent)
        assert state['best']['metrics']['official_fid'] == 28.75006367287608
        shutil.copy2(source_resume, latest)
        shutil.copy2(source_folder / 'selected.pt', selected)
        write(stage_folder / 'resume_source.json', {
            'source_full_resume': str(source_resume),
            'source_full_resume_sha256': trainer.shared.sha(source_resume),
            'source_selected': str(source_folder / 'selected.pt'),
            'source_selected_sha256': trainer.shared.sha(source_folder / 'selected.pt'),
            'source_update': 20000, 'source_best_update': state['best']['update'],
            'source_best_official_fid': state['best']['metrics']['official_fid'],
            'optimizer_and_rng_state_preserved': True,
        })
        del state


def main():
    global original_report
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text())
    policy = config['adaptive_policy']
    trainer.C = config
    trainer.CONFIG_PATH = CONFIG_PATH
    trainer.OUT = OUT
    trainer.DOC = OUT / 'progress.md'
    trainer.room = branch_room
    trainer.POLICY = policy
    trainer.POLICY_SHA = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    trainer.S.update({'status': 'waiting_for_exclusive_gpu_lock', 'phase': 'preflight', 'current_stage': None,
                      'stages': {}, 'started_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())})
    original_report = trainer.report
    trainer.report = mirror_report
    original_evaluate = trainer.evaluate_v
    def evaluate_and_prune_features(backbone, times, clip, folder, *args, **kwargs):
        metrics = original_evaluate(backbone, times, clip, folder, *args, **kwargs)
        (Path(folder) / 'features.npy').unlink(missing_ok=True)
        return metrics
    trainer.evaluate_v = evaluate_and_prune_features

    lock_path = EXP / 'gt_gan.lock'
    with lock_path.open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                trainer.S['status'] = 'waiting_for_exclusive_gpu_lock'
                trainer.S['phase'] = 'preflight'
                trainer.S['current_stage'] = None
                trainer.report()
                time.sleep(30)
        assert torch.cuda.is_available(), 'CUDA is required; no GPU experiment was started.'
        assert config['launch_authorized'] is True
        assert not (OUT / 'completed.json').exists()
        previous_failure = OUT / 'failure.json'
        if previous_failure.exists():
            archive = OUT / 'attempt_failures'
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(previous_failure), str(archive / f"failure_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.json"))
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        trainer.seed()
        assert trainer.shared.sha(Path(config['teacher']['checkpoint'])) == config['teacher']['sha256']
        teacher_state = torch.load(config['teacher']['checkpoint'], map_location='cpu', weights_only=False)
        trainer.student_config = {**teacher_state['config'], 'model_channels': 130,
                                  'prediction_type': 'v',
                                  'activation_checkpoint_resolutions': [64, 32, 16, 8]}
        del teacher_state
        trainer.alpha = Diffusion(1000, .0001, .02, torch.device('cuda')).alpha_bar
        trainer.dataset = None
        trainer.feature_net = None
        trainer.REVISION = repro.capture_revision(ROOT, OUT, CONFIG_PATH, Path(__file__))
        trainer.S.update({'status': 'running', 'phase': 'distillation', 'current_stage': None})
        write(OUT / 'experiment_plan.json', {
            'created_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'reason': 'User requested additional budget from N8 onward because Official FID may still improve.',
            'budgets': {'n8_total_updates': 45000, 'n4_total_updates': 60000, 'n2_total_updates': 75000},
            'selection_and_parent_rule': 'At every stage boundary, use the checkpoint with minimum measured Official FID; N4 starts from best N8 EMA, N2 starts from best N4 EMA.',
            'n8_resume': 'Continue optimizer/RNG state from the durable 20K N8 best Official-FID full-resume checkpoint; train to 45K.',
            'original_run': str(SOURCE), 'branch_output': str(OUT),
            'original_run_files_preserved': True,
            'storage_policy': '1.5 GiB minimum free space; omit redundant best full-state copy because latest full-state is checkpointed every 500 updates; retain selected best EMA, metrics, logs, and provenance. Delete per-evaluation Inception feature arrays as soon as each evaluation completes.',
            'config_sha256': trainer.shared.sha(CONFIG_PATH),
            'policy_sha256': trainer.POLICY_SHA,
            'source_revision': trainer.REVISION,
        })
        write(OUT / 'omit_best_full_resume.json', {
            'reason': 'Avoid duplicating 1.43 GiB optimizer checkpoints at every best-FID event. latest.pt is the durable resumable state, saved every 500 updates; selected.pt retains the best Official-FID EMA.',
            'latest_interval_updates': 500,
            'created_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        })
        write(OUT / 'config.json', config)
        parent = Path(json.loads((SOURCE / 'n8/config.json').read_text())['parent'])
        assert parent == SOURCE / 'n16/selected.pt'
        assert trainer.shared.sha(parent) == json.loads((SOURCE / 'n8/config.json').read_text())['parent_sha256']
        prepare_n8_resume(parent)
        trainer.seed()

        stages = [
            {'name': 'n8', 'updates': 45000, 'evaluation_interval': 5000, 'sampling_steps': 8},
            {'name': 'n4', 'updates': 60000, 'evaluation_interval': 5000, 'sampling_steps': 4},
            {'name': 'n2', 'updates': 75000, 'evaluation_interval': 5000, 'sampling_steps': 2},
        ]
        results = []
        for spec in stages:
            result = trainer.train_stage(spec, str(parent), 64)
            results.append(result)
            parent = Path(result['selected_checkpoint'])
            assert parent.is_file()
            write(OUT / 'chain.json', {
                'updated_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                'stages': results,
                'next_parent_checkpoint': str(parent),
                'next_parent_sha256': trainer.shared.sha(parent),
                'parent_rule': 'best Official FID selected EMA checkpoint',
            })
            stage_dir = OUT / spec['name']
            for feature_path in stage_dir.glob('eval_*/features.npy'):
                feature_path.unlink(missing_ok=True)
            # After a stage completes, preserve its selected EMA, scores, logs,
            # and provenance. The optimizer state is no longer needed for the
            # next stage and is removed to provide disk headroom.
            (stage_dir / 'latest.pt').unlink(missing_ok=True)
        trainer.S.update({'status': 'completed', 'phase': 'completed', 'current_stage': None})
        write(OUT / 'completed.json', {'status': trainer.S, 'stages': results,
                                      'final_selected_checkpoint': str(parent),
                                      'final_selected_checkpoint_sha256': trainer.shared.sha(parent)})
        trainer.report()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        trainer.S.update({'status': 'failed', 'error': __import__('traceback').format_exc()})
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / 'failure.json', trainer.S)
        try:
            trainer.report()
        except Exception:
            pass
        raise
