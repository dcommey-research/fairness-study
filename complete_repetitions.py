"""Schedule disjoint repetitions using the unchanged revision experiment function.

This driver filters only the outer seed enumeration; it does not modify folds,
fits, seeds, metrics, or the experiment hash. Driver provenance is separate.
"""
import argparse,builtins,hashlib,json,os,time
from pathlib import Path
from types import SimpleNamespace
from revision import run


def selected_enumerate(iterable,start=0,assigned=()):
    values=builtins.enumerate(iterable,start)
    if isinstance(iterable,list) and iterable==run.SEEDS:
        return ((i,v) for i,v in values if i in assigned)
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--repetition',type=int);p.add_argument('--pause-pid',type=int);a=p.parse_args()
    out=Path(a.output);manifest=json.loads((out/'manifest.json').read_text())
    if manifest['code_hash']!=run.code_hash():raise RuntimeError('Experiment hash mismatch')
    driver_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if a.pause_pid:
        import signal
        target=out/'folktables_income/native/n195665/r01-f05'
        while len(list(target.glob('*.json')))!=19:time.sleep(.1)
        os.kill(a.pause_pid,signal.SIGSTOP)
        run.save_json(out/'dispatch/coordinator.json',{'status':'paused_after_first_ACS_repetition','pid':a.pause_pid,'execution_driver_sha256':driver_hash,'utc':run.pd.Timestamp.now(tz='UTC').isoformat()})
        print('Coordinator paused after complete ACS repetition 1.',flush=True);return
    if a.repetition not in [2,3,4,5]:p.error('Worker repetitions must be 2..5')
    # Preserve repetition identifiers 1..5 while skipping unassigned seeds.
    assert list(selected_enumerate(run.SEEDS,1,[a.repetition]))==[(a.repetition,run.SEEDS[a.repetition-1])]
    assert list(selected_enumerate(iter(['a','b']),1,[a.repetition]))==[(1,'a'),(2,'b')]
    run.enumerate=lambda iterable,start=0:selected_enumerate(iterable,start,[a.repetition])
    params=SimpleNamespace(**manifest['args'])
    provenance={'repetition':a.repetition,'execution_driver_sha256':driver_hash,'experiment_code_sha256':manifest['code_hash'],'pid':os.getpid(),'started_utc':run.pd.Timestamp.now(tz='UTC').isoformat(),'status':'running'}
    run.save_json(out/f'dispatch/repetition-{a.repetition}.json',provenance)
    run.run_dataset('folktables_income',params)
    provenance.update(status='completed',completed_utc=run.pd.Timestamp.now(tz='UTC').isoformat())
    run.save_json(out/f'dispatch/repetition-{a.repetition}.json',provenance)

if __name__=='__main__':main()
