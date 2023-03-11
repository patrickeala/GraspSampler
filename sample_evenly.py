from graspsampler.GraspSampler import GraspSampler
import json
import argparse
from pathlib import Path
import numpy as np

# python sample_evenly.py --step 0.1 --extend 0.2 --quats_per_point 2 --category mug
# python sample_evenly.py --quats_per_point 14 --category mug

parser = argparse.ArgumentParser("Evenly sample grasp space.")
parser.add_argument("--trial", type=int, help="Trial.", default=1)
parser.add_argument("--extend", type=float, help="Limits of space in meteres.", default=0.35)
parser.add_argument("--step", type=float, help="Step size for grasp space slicing in meters", default=0.025)
parser.add_argument("--quats_per_point", type=int, help="Quaternion samples per each translation point.", default=0)
parser.add_argument(
    "--category",
    type=str,
    choices=["box", "cylinder", "mug", "bowl", "bottle", "hammer", "fork", "scissor"],
    help="Type of model to run.",
    required=True,
)
args = parser.parse_args()

extension = 'obj'
if (args.category == 'box') or (args.category == 'cylinder'):
    extension = 'stl'

def process(i):
    save_dir = 'grasp_data_generated'
    obj_filename = f'grasp_data/meshes/{args.category}/{args.category}{i:03}.{extension}'
    metadata_filename = f'grasp_data/info/{args.category}/{args.category}{i:03}.json'
    metadata = json.load(open(metadata_filename,'r'))

    # define grasp sampler
    graspsampler = GraspSampler(seed=args.trial)
    graspsampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])
    obj_pose_relative = graspsampler.obj.get_obj_mesh_mean()
    
    quaternions, translations, transforms, is_promising = graspsampler.sample_equal_spaced_grasps(extend=args.extend,
                                                                                            step=args.step,
                                                                                            quats_per_point=args.quats_per_point,
                                                                                            silent=True)

    
    main_save_dir = f'{save_dir}/{args.category}/{args.category}{i:03}'
    Path(main_save_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f'{main_save_dir}/main{args.trial}_even_grasps.npz',
                        transforms = transforms,
                        quaternions=quaternions,
                        translations=translations,
                        is_promising=is_promising,
                        obj_pose_relative=obj_pose_relative)

    print('------------------')
    print(f'Done for trial {args.trial}, category {args.category}, idx {i}')
    print(f'Total grasps are: {len(is_promising)}')
    print(f'Total promising candidates are: {np.sum(is_promising)}')

    del quaternions, translations, transforms, graspsampler

from joblib import Parallel, delayed
# n = 20
# if args.category == 'mug':
#     n = 21
Parallel(n_jobs=10)(delayed(process)(i) for i in range(20))