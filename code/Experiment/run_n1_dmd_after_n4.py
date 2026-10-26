"""DMD fine-tune the completed 90M N1 after N4 GT-GAN has finished."""
import copy
import fcntl
import hashlib
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path('/root/Diffusion 3')
EXP = ROOT / 'Experiment'
BASE = Path('/root/Diffusion/code/workspace_distillation')
OUT = ROOT / 'FID/student90_n1_dmd_after_n4_20261026'
N1_RUN = ROOT / 'FID/student90_n1_budget_extension_20261025'
CONFIG_PATH = EXP / 'configs/exp-20261021-student90-adaptive.json'
MAIN_DOC = ROOT / 'docs/STUDENT90_PROGRESS.md'
DOC = ROOT / 'docs/N1_DMD_PROGRESS.md'
sys.path[:0] = [str(EXP), str(EXP / 'source'), str(BASE)]

import student90_adaptive as trainer
import teacher_core
from teacher.cfg import apply_cfg
from teacher.diffusion import Diffusion
import adaptive_reproducibility as repro


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'python': random.getstate(), 'numpy': np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])


def seed_all():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)


def save_resume(path, state):
    if shutil.disk_usage(ROOT).free < 4 * 1024**3:
        raise RuntimeError('DMD safety stop: less than 4 GiB free before resume save.')
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temp)
    temp.replace(path)


def progress_report(original_report):
    def report_and_mirror():
        original_report()
        temp = MAIN_DOC.with_suffix('.tmp')
        temp.write_text(DOC.read_text(encoding='utf-8'), encoding='utf-8')
        temp.replace(MAIN_DOC)
    return report_and_mirror


def load_score_model(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['config']
    assert config['prediction_type'] == 'epsilon'
    model = teacher_core.model_from_config(config)
    weights = state.get('ema', state.get('state_dict'))
    assert weights is not None and state.get('weights', 'ema') == 'ema'
    model.load_state_dict(weights, strict=True)
    model = model.cuda().eval().requires_grad_(False)
    assert sum(p.numel() for p in model.parameters()) == 48_883_107
    return model, config, state


def main():
    assert torch.cuda.is_available(), 'CUDA is required for N1 DMD fine-tuning.'
    config = json.loads(CONFIG_PATH.read_text())
    trainer.C = config
    trainer.CONFIG_PATH = CONFIG_PATH
    trainer.OUT = OUT
    trainer.DOC = DOC
    trainer.POLICY = config['adaptive_policy']
    trainer.S = {'status': 'preflight', 'pid': __import__('os').getpid(),
                 'phase': 'waiting_for_n4_completion', 'current_stage': 'dmd_n1', 'stages': {}}
    trainer.REVISION = repro.capture_revision(ROOT, OUT, CONFIG_PATH, Path(__file__))
    trainer.seed()
    trainer.report = progress_report(trainer.report)
    OUT.mkdir(parents=True, exist_ok=True)

    # This runner is also safe to launch directly: wait for the N4 completion marker.
    n4_marker = N1_RUN / 'n4_extension_completed.json'
    while not n4_marker.exists():
        n4_status = json.loads((N1_RUN / 'status.json').read_text())
        if n4_status.get('status') == 'failed':
            raise RuntimeError('N4 GT-GAN continuation failed; refusing to start downstream DMD.')
        trainer.S.update({'status': 'waiting', 'phase': 'waiting_for_n4_completion'})
        trainer.report()
        time.sleep(30)

    n4_done = json.loads(n4_marker.read_text())
    n4_result = n4_done['result']
    n4_source = n4_result['best'].get('checkpoint') or n4_done['n4_baseline_checkpoint']
    n1_result = json.loads((N1_RUN / 'n1_result.json').read_text())
    n1_selected = Path(n1_result['selected_checkpoint'])
    (OUT / 'n1').mkdir(parents=True, exist_ok=True)
    n1_export = trainer.export_and_verify(n1_result)
    n4_model = trainer.Student90.load_checkpoint(n4_source, 'few_nfe', 'cpu')
    n4_digest = digest(n4_source)
    n1_model = trainer.Student90.load_checkpoint(n1_export, 'one_nfe', 'cpu')
    n1_digest = digest(n1_selected)
    assert n1_model.count_parameters() == 89_577_673
    assert n4_model.count_parameters() == 89_577_673
    assert len(n1_model.sampling_times) == 1 and len(n4_model.sampling_times) == 4
    del n1_model, n4_model

    teacher_path = Path(config['teacher']['checkpoint'])
    assert digest(teacher_path) == config['teacher']['sha256']
    real, real_config, _ = load_score_model(teacher_path)
    real_digest = digest(teacher_path)
    del _

    for stage in ['n8', 'n4', 'n2']:
        past = json.loads((ROOT / 'FID/student90_n8_n4_n2_budget_extension_20261023' / stage / 'result.json').read_text())
        trainer.S['stages'][stage] = {'status': 'completed', 'update': past['training_updates'],
            'budget': past['base_updates'], 'evaluations': past['evaluations'],
            'best_metrics': past['best']['metrics']}
    trainer.S['stages']['n1'] = {'status': 'completed', 'update': n1_result['training_updates'],
        'budget': n1_result['base_updates'], 'evaluations': n1_result['evaluations'],
        'best_metrics': n1_result['best']['metrics']}
    trainer.S['stages']['gan_n4'] = {'status': 'completed', 'update': 1000, 'budget': 1000,
        'evaluations': n4_result['evaluations'], 'best_metrics': n4_result['best']['metrics']}
    write(OUT / 'experiment_plan.json', {
        'method': 'DMD-inspired VP score-matching fine-tune with matched-noise N4 endpoint regression',
        'generator_initialization': str(n1_selected), 'generator_sha256': n1_digest,
        'endpoint_teacher': n4_source, 'endpoint_teacher_sha256': n4_digest,
        'real_score_teacher': str(teacher_path), 'real_score_sha256': real_digest,
        'generator_parameters': 89_577_673, 'score_model_parameters_each': 48_883_107,
        'generator_prediction': 'v', 'score_prediction': 'epsilon', 'dmd_cfg': 2.0,
        'generator_lr': 2e-5, 'fake_score_lr': 2e-4, 'ema_decay': 0.999,
        'effective_batch': 64, 'microbatch': 16, 'fake_score_warmup': 200,
        'noise_t_range': [50, 950], 'matched_endpoint_weight': 0.1,
        'screening_updates': 2500, 'continuation_policy': 'to 5000 only if Valid FID beats baseline; to 10000 only if 5000 beats 2500',
        'selection': 'minimum Valid FID including unchanged N1 baseline; Official FID is reporting only',
        'created_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    })

    student_state = torch.load(n1_selected, map_location='cpu', weights_only=False)
    student_config = student_state['student_config']
    student_times = student_state['sampling_times']
    clip = student_state['clip_denoised']
    student = trainer.Student90(student_config, student_times, clip).cuda()
    student.backbone.load_state_dict(student_state['ema'], strict=True)
    del student_state
    student.train().requires_grad_(True)
    ema = copy.deepcopy(student).eval().requires_grad_(False)
    endpoint = trainer.Student90.load_checkpoint(n4_source, 'few_nfe', 'cuda').eval().requires_grad_(False)
    fake = copy.deepcopy(real).train().requires_grad_(True)
    assert sum(p.numel() for p in student.parameters()) == 89_577_673

    g_optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5, betas=(0.9, 0.999), weight_decay=0)
    fake_optimizer = torch.optim.AdamW(fake.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=0)
    trainer.alpha = Diffusion(1000, real_config['beta_start'], real_config['beta_end'], torch.device('cuda')).alpha_bar
    g_updates = warmup = 0
    evaluations = []
    best = None
    folder = OUT / 'dmd_n1'
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / 'latest.pt'
    milestones = {2500, 5000, 10000}

    def update_fake(images, labels):
        fake_optimizer.zero_grad(set_to_none=True)
        times = torch.randint(50, 951, (len(labels),), device='cuda')
        a, sigma = trainer.coefficients(trainer.alpha, times)
        noise = torch.randn_like(images)
        noised = a * images.detach() + sigma * noise
        with trainer.amp():
            prediction = fake(noised, times, labels)
            loss = (prediction.float() - noise).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fake.parameters(), 1, error_if_nonfinite=True)
        fake_optimizer.step()
        return float(loss.detach())

    def student_image(noise, labels):
        return student.rollout(noise, labels)

    def fake_state():
        return {'student': student.state_dict(), 'ema': ema.state_dict(), 'fake': fake.state_dict(),
                'g_optimizer': g_optimizer.state_dict(), 'fake_optimizer': fake_optimizer.state_dict(),
                'g_updates': g_updates, 'warmup': warmup, 'evaluations': evaluations, 'best': best,
                'rng': rng_state(), 'n1_parent': str(n1_selected), 'n1_sha256': n1_digest,
                'n4_parent': n4_source, 'n4_sha256': n4_digest, 'real_score': str(teacher_path),
                'real_score_sha256': real_digest}

    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        assert state['n1_sha256'] == n1_digest and state['n4_sha256'] == n4_digest
        assert state['real_score_sha256'] == real_digest
        student.load_state_dict(state['student']); ema.load_state_dict(state['ema']); fake.load_state_dict(state['fake'])
        g_optimizer.load_state_dict(state['g_optimizer']); fake_optimizer.load_state_dict(state['fake_optimizer'])
        g_updates, warmup = state['g_updates'], state['warmup']
        evaluations, best = state['evaluations'], state['best']
        restore_rng(state['rng']); del state

    trainer.S.update({'status': 'running', 'phase': 'preflight', 'current_stage': 'dmd_n1'})
    trainer.S['stages']['dmd_n1'] = {'status': 'preflight', 'update': g_updates, 'budget': 10000,
                                     'evaluations': evaluations, 'best_metrics': (best or {}).get('metrics', {})}
    trainer.report()
    if not (folder / 'preflight.json').exists():
        rng = rng_state()
        labels = torch.tensor([0, 47], device='cuda')
        noise = torch.randn((2, 3, 64, 64), device='cuda', generator=torch.Generator(device='cuda').manual_seed(42))
        with torch.no_grad(), trainer.amp():
            s1 = student_image(noise, labels)
            s4 = endpoint.rollout(noise, labels)
        assert s1.shape == s4.shape and torch.isfinite(s1).all() and torch.isfinite(s4).all()
        times = torch.full_like(labels, 500)
        a, sigma = trainer.coefficients(trainer.alpha, times)
        probe = a * s1 + sigma * torch.randn_like(s1)
        with torch.no_grad(), trainer.amp():
            uncond = real(probe, times, torch.full_like(labels, 151))
            cond = real(probe, times, labels)
            score = apply_cfg(uncond, cond, 2.0)
        assert score.shape == probe.shape and torch.isfinite(score).all()
        write(folder / 'preflight.json', {'n1_nfe': 1, 'n4_nfe': 4, 'n1_parameters': 89_577_673,
              'real_score_parameters': 48_883_107, 'teacher_prediction': 'epsilon', 'dmd_cfg': 2.0,
              'matched_endpoint_noise_shape': list(noise.shape), 'dmd_gradient_finite': True,
              'n1_sha256': n1_digest, 'n4_sha256': n4_digest, 'real_score_sha256': real_digest})
        restore_rng(rng)

    if best is None:
        trainer.S['phase'] = 'baseline_evaluation'
        metrics = trainer.evaluate_v(ema.backbone, ema.sampling_times, ema.clip_denoised, folder / 'eval_000000')
        evaluations.append({'update': 0, **metrics})
        best = {'update': 0, 'metrics': metrics, 'checkpoint': n1_export}
        trainer.S['stages']['dmd_n1'].update({'evaluations': evaluations, 'best_metrics': metrics})
        trainer.report()

    # Historical recipe: 200 fake-score warmup updates; no LR warmup.
    while warmup < 200:
        labels = torch.randint(151, (16,), device='cuda')
        noise = torch.randn((16, 3, 64, 64), device='cuda')
        with torch.no_grad(), trainer.amp():
            images = student_image(noise, labels)
        fake_loss = update_fake(images, labels)
        warmup += 1
        if warmup % 20 == 0:
            rec = {'warmup': warmup, 'fake_score_loss': fake_loss}
            with (folder / 'train.jsonl').open('a') as f:
                f.write(json.dumps({'phase': 'fake_warmup', **rec}) + '\n')
            trainer.S['stages']['dmd_n1'].update({'status': 'fake_score_warmup', 'warmup': warmup,
                                                  'last_log': rec})
            trainer.report()
        if warmup % 50 == 0:
            save_resume(latest, fake_state())

    while g_updates < 10000:
        start = time.monotonic()
        g_optimizer.zero_grad(set_to_none=True)
        dmd_losses, endpoint_losses, fake_losses = [], [], []
        for micro in range(4):
            labels = torch.randint(151, (16,), device='cuda')
            seed = 1_000_000 + g_updates * 4 + micro
            z = torch.randn((16, 3, 64, 64), device='cuda',
                            generator=torch.Generator(device='cuda').manual_seed(seed))
            with torch.no_grad(), trainer.amp():
                detached = student_image(z, labels)
            fake_loss = update_fake(detached, labels)
            times = torch.randint(50, 951, (16,), device='cuda')
            a, sigma = trainer.coefficients(trainer.alpha, times)
            eps = torch.randn_like(detached)
            noised = a * detached + sigma * eps
            with torch.no_grad(), trainer.amp():
                uncond = real(noised, times, torch.full_like(labels, 151))
                cond = real(noised, times, labels)
                real_eps = apply_cfg(uncond, cond, 2.0).float()
                fake_eps = fake(noised, times, labels).float()
                real_x0 = (noised - sigma * real_eps) / a
                fake_x0 = (noised - sigma * fake_eps) / a
                norm = (detached - real_x0).abs().mean((1, 2, 3), keepdim=True).clamp_min(0.001)
                distribution_gradient = (fake_x0 - real_x0) / norm
                endpoint_target = endpoint.rollout(z, labels)
            assert torch.isfinite(distribution_gradient).all()
            generated = student_image(z, labels)
            dmd_target = (generated.detach() - distribution_gradient).detach()
            dmd_loss = 0.5 * (generated - dmd_target).square().mean()
            endpoint_loss = (generated - endpoint_target).square().mean()
            loss = (dmd_loss + 0.1 * endpoint_loss) / 4
            assert torch.isfinite(loss)
            loss.backward()
            dmd_losses.append(float(dmd_loss.detach()))
            endpoint_losses.append(float(endpoint_loss.detach()))
            fake_losses.append(fake_loss)

        grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1, error_if_nonfinite=True))
        g_optimizer.step()
        with torch.no_grad():
            for target, param in zip(ema.parameters(), student.parameters(), strict=True):
                target.lerp_(param.detach(), 0.001)
        g_updates += 1
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        rec = {'update': g_updates, 'dmd_loss': float(np.mean(dmd_losses)),
               'endpoint_loss': float(np.mean(endpoint_losses)), 'fake_score_loss': float(np.mean(fake_losses)),
               'gradient_norm': grad_norm, 'update_seconds': elapsed,
               'peak_vram_bytes': torch.cuda.max_memory_allocated()}
        with (folder / 'train.jsonl').open('a') as f:
            f.write(json.dumps(rec) + '\n')
        if g_updates % 20 == 0 or g_updates == 1:
            trainer.S['phase'] = 'dmd_training'
            trainer.S['stages']['dmd_n1'].update({'status': 'training', 'update': g_updates,
                'budget': 10000, 'warmup': warmup, 'last_log': rec, 'evaluations': evaluations,
                'best_metrics': best['metrics']})
            trainer.report()
            print(json.dumps(rec), flush=True)
        if g_updates % 100 == 0 or g_updates in milestones:
            save_resume(latest, fake_state())
        if g_updates in milestones:
            trainer.S['phase'] = 'evaluation'
            metrics = trainer.evaluate_v(ema.backbone, ema.sampling_times, ema.clip_denoised,
                                         folder / f'eval_{g_updates:06d}')
            evaluations.append({'update': g_updates, **metrics})
            # Select on the documented development/Valid split, never Official FID.
            if metrics['valid_fid'] < best['metrics']['valid_fid']:
                ckpt = folder / f'dmd_n1_{g_updates:06d}.ckpt'
                trainer.export(ckpt, ema.backbone, student_config, ema.sampling_times,
                               ema.clip_denoised, {'source_n1': str(n1_selected), 'source_n4': n4_source,
                               'dmd_update': g_updates, 'metrics': metrics})
                best = {'update': g_updates, 'metrics': metrics, 'checkpoint': str(ckpt)}
            trainer.S['stages']['dmd_n1'].update({'status': 'training', 'update': g_updates,
                'budget': 10000, 'evaluations': evaluations, 'best_metrics': best['metrics']})
            trainer.report()
            if g_updates == 2500 and metrics['valid_fid'] >= evaluations[0]['valid_fid']:
                break
            if g_updates == 5000 and metrics['valid_fid'] >= next(
                    e['valid_fid'] for e in evaluations if e['update'] == 2500):
                break

    # If no DMD checkpoint beat the input, retain the unchanged N1 baseline.
    selected = best['checkpoint']
    if best['update'] == 0:
        selected = n1_export
    result = {'stage': 'dmd_n1', 'source_n1': str(n1_selected), 'source_n1_sha256': n1_digest,
              'source_n4': n4_source, 'source_n4_sha256': n4_digest,
              'real_score': str(teacher_path), 'real_score_sha256': real_digest,
              'training_updates': g_updates, 'fake_score_warmup_updates': warmup,
              'evaluations': evaluations, 'selected_by': 'minimum Valid FID including unchanged N1 baseline',
              'best': best, 'selected_checkpoint': selected, 'selected_checkpoint_sha256': digest(selected),
              'parameters': 89_577_673, 'nfe': 1, 'dmd_cfg': 2.0,
              'recipe': 'DMD normalized x0 score difference + 0.1 matched-noise N4 endpoint MSE'}
    write(folder / 'result.json', result)
    write(OUT / 'completed.json', result)
    latest.unlink(missing_ok=True)
    trainer.S.update({'status': 'completed', 'phase': 'completed', 'current_stage': None})
    trainer.S['stages']['dmd_n1'].update({'status': 'completed', 'update': g_updates,
        'budget': 10000, 'evaluations': evaluations, 'best_metrics': best['metrics']})
    trainer.report()
    shutil.copyfile(DOC, ROOT / 'docs/N1_DMD_RESULTS.md')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / 'failure.json', {'status': 'failed', 'traceback': traceback.format_exc(),
              'updated_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())})
        raise
