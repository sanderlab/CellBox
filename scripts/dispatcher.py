"""
This is the grid search script which would dispatch multiple jobs
"""
import json
import argparse
import pdb
import os
import shutil


parser = argparse.ArgumentParser(description='CellBox grid search dispatcher')
parser.add_argument('--grid_config_path', required=True, type=str, help="Path of grid search config")
parser.add_argument('--meta_config_path', required=True, type=str, help="Path of meta search config")
parser.add_argument('--no_submission', action='store_true', help='whether to submit jobs to O2')
parser.add_argument('--grid_name', required=False, default='grid', type=str, help="Name of the current grid search")
args = parser.parse_args()

grid = json.load(open(args.grid_config_path, 'r'))
meta_cfg = json.load(open(args.meta_config_path, 'r'))


def append(parent_job, new_handle):
    job = parent_job.copy()
    job.update(new_handle)
    return job


modifiers = [{}]
barcodes = ['']
for key in grid:
    modifiers = [append(job, {key: val}) for job in modifiers for val in grid[key]]
    barcodes = [barcode + '_{}'.format(i)  for barcode in barcodes for i, _ in enumerate(grid[key])]

wdr = os.path.join(os.path.dirname(args.meta_config_path), 'grid_search')
bash_file = os.path.join(os.path.dirname(args.meta_config_path), 'run.sh')
if os.path.exists(wdr):
    shutil.rmtree(wdr)
os.makedirs(wdr)
for modifier, barcode in zip(modifiers, barcodes):
    job = meta_cfg.copy()
    job.update(modifier)
    job.update({'experiment_id': job['experiment_id'] + '_' + args.grid_name + barcode})
    job_cfg_path = os.path.join(wdr, args.grid_name + barcode + '.json')
    json.dump(job, open(job_cfg_path, 'w'), indent=4, sort_keys=True)
    # for i in range(args.duplicates):
    #     print('python scripts/main.py --experiment_config_path={} --working_index={}'.format(job_cfg_path, i))
    if not args.no_submission:
        os.system("sbatch {} {}".format(bash_file, job_cfg_path))
    else:
        print("sbatch {} {}".format(bash_file, job_cfg_path))

