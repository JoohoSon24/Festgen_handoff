"""Continue CFG internalization from its best full-state checkpoint.

This is a separately recorded branch. It retains the parent run and uses the
same training recipe, optimizer state, data manifest, and evaluation seeds.
"""
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path('/root/Diffusion 3')
BASE = Path('/root/Diffusion/code/workspace_distillation')
EXP = ROOT / 'Experiment'
PARENT_RUN = ROOT / 'FID/student90_cfg25_restart17'
OUT = ROOT / 'FID/student90_guidance32_extension_20261022'
DOC = ROOT / 'docs/STUDENT90_INTERNALIZATION_EXTENSION.md'
MAIN_DOC = ROOT / 'docs/STUDENT90_PROGRESS.md'
CONFIG_PATH = EXP / 'configs/exp-20261021-student90-adaptive.json'
SOURCE_RESUME = PARENT_RUN / 'guidance32/best_resume.pt'
SOURCE_SELECTED = PARENT_RUN / 'guidance32/selected.pt'
SOURCE_CONFIG = EXP / 'configs/exp-20261021-student90-adaptive.json'

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


def build_plot():
    status = json.loads((PARENT_RUN / 'status.json').read_text())
    rows = list(status['stages']['guidance32']['evaluations'])
    extension = OUT / 'guidance32'
    for path in sorted(extension.glob('eval_*/metrics.json')):
        update = int(path.parent.name.removeprefix('eval_'))
        metric = json.loads(path.read_text())
        rows = [row for row in rows if row['update'] != update]
        rows.append({'update': update, **metric})
    rows.sort(key=lambda x: x['update'])
    plot = ROOT / 'docs/STUDENT90_INTERNALIZATION_TREND.svg'
    width, height = 1100, 560
    left, top, pw, ph = 88, 105, 940, 330
    vals = [float(row['official_fid']) for row in rows]
    ymin, ymax = max(0, min(vals) - 2), max(vals) + 3
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:25px;font-weight:bold}.axis{font-size:13px}.grid{stroke:#d9e0ea}.line{fill:none;stroke:#2878b5;stroke-width:3}.best{stroke:#e4572e;stroke-width:2;stroke-dasharray:7 5}.mark{fill:#2878b5}</style>',
             '<text x="40" y="42" class="title">90M student — CFG internalization Official FID (NFE 32)</text>',
             '<text x="40" y="73" class="axis">Same fixed 3020-image evaluation set and seed schedule at every checkpoint</text>']
    for i in range(5):
        y = top + ph * i / 4
        value = ymax - (ymax - ymin) * i / 4
        parts += [f'<line x1="{left}" y1="{y}" x2="{left+pw}" y2="{y}" class="grid"/>',
                  f'<text x="38" y="{y+4}" class="axis">{value:.1f}</text>']
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#333"/>',
              f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>']
    xmax = max(250000, max(row['update'] for row in rows))
    for tick in range(0, xmax + 1, 50000):
        x = left + pw * tick / xmax
        parts += [f'<line x1="{x}" y1="{top+ph}" x2="{x}" y2="{top+ph+5}" stroke="#333"/>',
                  f'<text x="{x-15}" y="{top+ph+24}" class="axis">{tick//1000}k</text>']
    coords = []
    for row in rows:
        x = left + pw * row['update'] / xmax
        y = top + ph * (ymax - float(row['official_fid'])) / (ymax - ymin)
        coords.append((x, y, row))
    points = ' '.join(f'{x:.1f},{y:.1f}' for x, y, _ in coords)
    parts.append(f'<polyline points="{points}" class="line"/>')
    best = min(rows, key=lambda row: row['official_fid'])
    bx = left + pw * best['update'] / xmax
    by = top + ph * (ymax - best['official_fid']) / (ymax - ymin)
    parts.append(f'<line x1="{left}" y1="{by}" x2="{left+pw}" y2="{by}" class="best"/>')
    for x, y, row in coords:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="mark"><title>{row["update"]}: {row["official_fid"]:.6f}</title></circle>')
    parts.append(f'<text x="{min(bx+9,left+pw-180):.1f}" y="{by-9:.1f}" class="axis">Best {best["official_fid"]:.4f} @ {best["update"]//1000}k</text>')
    parts += [f'<text x="{left+pw/2-55}" y="{top+ph+51}" class="axis">Training updates</text>',
              '<text x="40" y="530" class="axis">Separate NFE-16 early result: step 0 Official 30.164 / Valid 46.779; step 5K Official 27.607 / Valid 46.403.</text>', '</svg>']
    plot.write_text('\n'.join(parts), encoding='utf-8')
    csv = ROOT / 'docs/STUDENT90_INTERNALIZATION_TREND.csv'
    with csv.open('w', encoding='utf-8') as stream:
        stream.write('stage,update,official_fid,valid_fid,precision,recall\n')
        for row in rows:
            stream.write(f'guidance32,{row["update"]},{row["official_fid"]:.9f},{row["valid_fid"]:.9f},{row.get("precision_k3", float("nan")):.9f},{row.get("recall_k3", float("nan")):.9f}\n')


def main():
    assert torch.cuda.is_available(), 'CUDA is required for this continuation.'
    assert SOURCE_RESUME.is_file() and SOURCE_SELECTED.is_file()
    assert not (OUT / 'completed.json').exists()
    config = json.loads(SOURCE_CONFIG.read_text())
    policy = config['adaptive_policy']
    OUT.mkdir(parents=True, exist_ok=True)
    trainer.C = config
    trainer.CONFIG_PATH = CONFIG_PATH
    trainer.OUT = OUT
    trainer.DOC = DOC
    trainer.POLICY = policy
    trainer.POLICY_SHA = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    trainer.S.update({'status': 'running', 'phase': 'guidance32_extension', 'current_stage': 'guidance32_extension', 'stages': {}, 'started_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())})
    (OUT / 'guidance32').mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_RESUME, OUT / 'guidance32/latest.pt')
    shutil.copy2(SOURCE_RESUME, OUT / 'guidance32/best_resume.pt')
    shutil.copy2(SOURCE_SELECTED, OUT / 'guidance32/selected.pt')
    (OUT / 'teacher32_anchor').mkdir(parents=True, exist_ok=True)
    shutil.copy2(PARENT_RUN / 'teacher32_anchor/metrics.json', OUT / 'teacher32_anchor/metrics.json')
    # Keep the original approved config and its policy hash; only the stage target
    # is raised to the already authorized 250K cap in the call below.
    state = torch.load(OUT / 'guidance32/latest.pt', map_location='cpu', weights_only=False)
    old_config_sha = state['config_sha256']
    update = int(state['update'])
    assert update == 210000, f'Expected the preserved best full-state checkpoint at 210K, got {update}.'
    shutil.copy2(SOURCE_CONFIG, OUT / 'provenance_source_config.json')
    trainer.student_config = state['student_config']
    trainer.alpha = Diffusion(1000, .0001, .02, torch.device('cuda')).alpha_bar
    trainer.dataset = None
    trainer.feature_net = None
    trainer.REVISION = repro.capture_revision(ROOT, OUT, CONFIG_PATH, Path(__file__))
    write(OUT / 'source_resume_record.json', {
        'source_run': str(PARENT_RUN), 'source_checkpoint': str(SOURCE_RESUME),
        'source_sha256': trainer.shared.sha(SOURCE_RESUME), 'resume_update': update,
        'source_best_official_fid': state['best']['metrics']['official_fid'],
        'weights': 'raw+EMA+AdamW+RNG+epoch cursor',
        'additional_updates_cap': 40000, 'effective_batch': 64,
        'original_config_sha256': trainer.shared.sha(CONFIG_PATH),
        'checkpoint_config_sha256': old_config_sha,
        'policy_sha256': trainer.POLICY_SHA,
        'objective': config['training']['objective'],
        'seed': config['training']['seed'], 'train_manifest_sha256': config['training']['train_manifest_sha256'],
    })
    # The checkpoint already carries the measured history and best EMA at 210K.
    write(OUT / 'continuation_plan.json', {
        'stage': 'guidance32', 'resume_update': update, 'target_update': 250000,
        'additional_updates': 40000, 'evaluation_interval': 5000,
        'reason': 'User requested more CFG internalization; use remaining bounded guidance32 cap.',
        'selection': 'best Official FID among measured EMA checkpoints; existing main run preserved.',
        'created_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    })
    write(OUT / 'guidance32/adaptive_start.json', {
        'update': update, 'best': state['best'], 'source_checkpoint_sha256': trainer.shared.sha(SOURCE_RESUME),
        'continuation': True, 'policy_sha256': trainer.POLICY_SHA,
    })
    del state
    original_report = trainer.report
    def report_with_plot():
        original_report()
        build_plot()
        main_tmp = MAIN_DOC.with_suffix('.tmp')
        main_tmp.write_text(DOC.read_text(encoding='utf-8'), encoding='utf-8')
        main_tmp.replace(MAIN_DOC)
    trainer.report = report_with_plot
    trainer.seed()
    trainer.log_event('guidance32_extension', status='starting', update=update, budget=250000)
    result = trainer.train_stage({'name': 'guidance32', 'updates': 250000, 'evaluation_interval': 5000, 'sampling_steps': 32}, config['teacher']['checkpoint'], 64)
    write(OUT / 'completed.json', {'status': 'completed', 'result': result, 'source_resume': str(SOURCE_RESUME)})
    trainer.S.update({'status': 'completed', 'phase': 'completed', 'current_stage': None})
    trainer.report()
    build_plot()


if __name__ == '__main__':
    main()
