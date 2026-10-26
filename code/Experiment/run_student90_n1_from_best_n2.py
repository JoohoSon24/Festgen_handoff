"""Distill N1 from best N2 EMA, then GT-GAN fine-tune the selected N4 EMA."""
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
SOURCE_RUN = ROOT / 'FID/student90_n8_n4_n2_budget_extension_20261023'
OUT = ROOT / 'FID/student90_n1_budget_extension_20261025'
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


def branch_room():
    free = shutil.disk_usage(ROOT).free
    floor = 2 * 1024**3
    if free < floor:
        raise RuntimeError(f'Disk free {free / 1024**3:.2f} GiB < N1 safety floor 2.0 GiB.')


def mirror_report():
    original_report()
    temp = MAIN_DOC.with_suffix('.tmp')
    temp.write_text(trainer.DOC.read_text(encoding='utf-8'), encoding='utf-8')
    temp.replace(MAIN_DOC)


def completed_summary(name):
    result = json.loads((SOURCE_RUN / name / 'result.json').read_text())
    logs = (SOURCE_RUN / name / 'train.jsonl').read_text().splitlines()
    return {
        'status': 'completed', 'update': result['training_updates'],
        'budget': result['base_updates'], 'evaluations': result['evaluations'],
        'best_metrics': result['best']['metrics'],
        'last_log': json.loads(logs[-1]) if logs else {},
    }


def main():
    global original_report
    assert torch.cuda.is_available(), 'CUDA is required for N1 distillation.'
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text())
    policy = config['adaptive_policy']
    trainer.C = config
    trainer.CONFIG_PATH = CONFIG_PATH
    trainer.OUT = OUT
    trainer.DOC = DOC
    trainer.room = branch_room
    trainer.POLICY = policy
    trainer.POLICY_SHA = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    trainer.S.update({
        'status': 'waiting_for_exclusive_gpu_lock', 'phase': 'preflight',
        'current_stage': None, 'stages': {
            name: completed_summary(name) for name in ['n8', 'n4', 'n2']
        }, 'started_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    })
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
                trainer.report()
                time.sleep(30)
        assert torch.cuda.is_available()
        assert config['launch_authorized'] is True
        assert not (OUT / 'completed.json').exists()
        old_failure = OUT / 'failure.json'
        if old_failure.exists():
            archive = OUT / 'attempt_failures'
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_failure), str(archive / f"failure_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.json"))
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
        trainer.S.update({'status': 'running', 'phase': 'distillation'})
        write(OUT / 'omit_best_full_resume.json', {
            'reason': 'Keep only selected best Official-FID EMA as stage output; latest full state is saved every 500 updates during active N1 training and then removed after completion. FID metrics and train logs remain.',
            'latest_interval_updates': 500,
            'created_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        })
        write(OUT / 'experiment_plan.json', {
            'created_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'stage': 'n1', 'total_updates': 112500,
            'reason': 'N1 is the next progressive-distillation stage after completed N8→N4→N2; then apply the previously validated GT conditional hinge GAN + frozen-anchor N4 fine-tune (no DMD).',
            'parent_checkpoint': str(SOURCE_RUN / 'n2/selected.pt'),
            'parent_official_fid': 43.252674965719166,
            'parent_rule': 'best Official-FID EMA checkpoint from N2',
            'source_chain_sha256': trainer.shared.sha(SOURCE_RUN / 'chain.json'),
            'config_sha256': trainer.shared.sha(CONFIG_PATH),
            'policy_sha256': trainer.POLICY_SHA,
            'source_revision': trainer.REVISION,
            'followup': {'stage': 'n4 GT-GAN fine-tune', 'source': str(SOURCE_RUN / 'n4/selected.pt'), 'steps': 1000, 'evaluation_updates': [250, 500, 1000], 'dmd': False},
        })
        write(OUT / 'config.json', config)
        parent = SOURCE_RUN / 'n2/selected.pt'
        n2_result = json.loads((SOURCE_RUN / 'n2/result.json').read_text())
        chain = json.loads((SOURCE_RUN / 'chain.json').read_text())
        assert str(parent) == n2_result['selected_checkpoint'] == chain['next_parent_checkpoint'] and parent.is_file()
        assert trainer.shared.sha(parent) == chain['next_parent_sha256']
        trainer.seed()
        spec = {'name': 'n1', 'updates': 112500, 'evaluation_interval': 5000, 'sampling_steps': 1}
        result = trainer.train_stage(spec, str(parent), 64)
        stage_dir = OUT / 'n1'
        for feature_path in stage_dir.glob('eval_*/features.npy'):
            feature_path.unlink(missing_ok=True)
        # Keep the optimal EMA and all FID/loss records; discard nonselected optimizer state.
        (stage_dir / 'latest.pt').unlink(missing_ok=True)
        write(OUT / 'n1_result.json', result)
        n4_result = json.loads((SOURCE_RUN / 'n4/result.json').read_text())
        n4_baseline = trainer.export_and_verify(n4_result)
        gan_result = trainer.gan_finetune('n4', n4_baseline)
        gan_folder = OUT / 'gan_n4'
        for feature_path in gan_folder.glob('eval_*/features.npy'):
            feature_path.unlink(missing_ok=True)
        (gan_folder / 'latest.pt').unlink(missing_ok=True)
        trainer.S.update({'status': 'completed', 'phase': 'completed', 'current_stage': None})
        write(OUT / 'completed.json', {
            'status': trainer.S, 'n1_distillation': result, 'n4_gtgan_finetune': gan_result,
            'n1_selected_checkpoint': result['selected_checkpoint'],
            'n1_best_official_fid': result['best']['metrics']['official_fid'],
            'n1_parent_checkpoint': str(parent), 'n1_parent_sha256': trainer.shared.sha(parent),
            'n4_baseline_checkpoint': n4_baseline,
            'n4_gan_selected_checkpoint': gan_result['best'].get('checkpoint'),
            'n4_gan_best_official_fid': gan_result['best']['metrics']['official_fid'],
        })
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
