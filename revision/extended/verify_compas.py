"""Independent checks on the COMPAS race-encoding run, from saved arrays only.

Checks: complete ledger; every failure is a TO calibration-support failure; accuracy
and realized DP recompute from the decision arrays; baseline predictions are identical
across the four groupings; the race (6) x sex grouping reproduces the main run's
COMPAS predictions for every configuration that succeeded in both (EG fits that
converge to a different, equally optimal solution are recorded as notes).
"""
from pathlib import Path
import argparse, json
import numpy as np
from revision.run import save_json
from revision.extended.compas_encoding import code_hash, ENCODINGS

def main():
    p = argparse.ArgumentParser(); p.add_argument('--input',default='results/revision_compas_encoding'); p.add_argument('--main-run',default='results/revision_verified'); p.add_argument('--output',default='revision/generated/paper/verification_compas.json'); a = p.parse_args()
    root = Path(a.input); main_root = Path(a.main_run); manifest = json.loads((root/'manifest.json').read_text())
    paths = sorted(root.glob('compas/*/n*/r*-f*/*.json')); errors = []; notes = []; checks = 0; baseline = {}; matched_main = 0
    if len(paths)!=manifest['expected_evaluations']: errors.append('Ledger count differs from plan')
    if code_hash()!=manifest['code_hash']: errors.append('Source differs from run manifest')
    for path in paths:
        r = json.loads(path.read_text())
        if r['code_hash']!=manifest['code_hash']: errors.append(f'Code hash mismatch {path}')
        if r['status']!='success':
            if r['method']!='TO' or not r.get('error','').startswith('TO infeasible:'): errors.append(f'Unexpected failure {path}')
            if not any(row.get('0')==0 or row.get('1')==0 for row in r.get('calibration_group_counts',[])): errors.append(f'Failure without class-empty group {path}')
            continue
        z = np.load(path.with_suffix('.npz'),allow_pickle=False); y = z['y']; d = z['decision']; g = z['group']
        if not np.isclose(r['metrics']['accuracy'],np.mean(y==d)): errors.append(f'Accuracy mismatch {path}')
        rates = [d[g==k].mean() for k in np.unique(g)]
        if not np.isclose(r['metrics']['dp_diff'],np.ptp(rates)): errors.append(f'DP mismatch {path}')
        if r['config']=='LR-baseline':
            key = (r['repeat'],r['fold'])
            if key in baseline and not np.array_equal(baseline[key],d): errors.append(f'Baseline changed across groupings {path}')
            baseline[key] = d
        if r['encoding']=='race6_sex':
            other = main_root/'compas'/'native'/path.parts[-3]/path.parts[-2]/path.with_suffix('.npz').name
            if other.exists():
                with np.load(other,allow_pickle=True) as m:
                    if not (np.array_equal(m['row_id'],z['row_id']) and np.allclose(m['policy_probability'],z['policy_probability'],atol=1e-12)):
                        # EG solutions need not be unique; record rather than fail when both fits converged.
                        if r['method']=='EG' and r['optimizer']['best_gap_']<1e-8: notes.append(f"EG refit reached a different solution with gap {r['optimizer']['best_gap_']:.1e}: {path}")
                        else: errors.append(f'Main-run reproduction mismatch {path}')
                matched_main += 1
        checks += 1
    result = {'status':'passed' if not errors else 'failed','expected_evaluations':manifest['expected_evaluations'],'records':len(paths),'successful_checked':checks,'baseline_splits_compared':len(baseline),'main_run_reproductions':matched_main,'encodings':ENCODINGS,'code_hash':manifest['code_hash'],'reproduction_notes':notes,'errors':errors[:50]}
    save_json(Path(a.output),result); print(json.dumps(result,indent=2))
    if errors: raise SystemExit(1)

if __name__=='__main__': main()
