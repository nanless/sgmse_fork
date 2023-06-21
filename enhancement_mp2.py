import glob
from argparse import ArgumentParser
from os.path import join
import os

import torch
import torch.multiprocessing as mp
from soundfile import write
from torchaudio import load
from tqdm import tqdm

from sgmse.model import ScoreModel
from sgmse.util.other import ensure_dir, pad_spec


def process_file(task):
    gpu_id = task['gpu_id']

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(gpu_id)

    # Load score model
    model = ScoreModel.load_from_checkpoint(
        task['checkpoint_file'], base_dir='', batch_size=16, num_workers=0,
        kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()

    filename = task['noisy_file'].split('/')[-1]

    # Load wav
    y, _ = load(task['noisy_file'])
    T_orig = y.size(1)

    # Normalize
    norm_factor = y.abs().max()
    y = y / norm_factor

    # Prepare DNN input
    Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
    Y = pad_spec(Y)

    # Reverse sampling
    sampler = model.get_pc_sampler(
        'reverse_diffusion', task['corrector_cls'], Y.cuda(), N=task['N'],
        corrector_steps=task['corrector_steps'], snr=task['snr'])
    sample, _ = sampler()

    # Backward transform in time domain
    x_hat = model.to_audio(sample.squeeze(), T_orig)

    # Renormalize
    x_hat = x_hat * norm_factor

    # Write enhanced wav file
    write(join(task['target_dir'], filename), x_hat.cpu().numpy(), 16000)

    del model
    del sampler
    del sample
    del x_hat
    del Y
    del y
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True,
                        help='Directory containing the test data (must have subdirectory noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True,
                        help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald",
                        help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    args = parser.parse_args()

    noisy_dir = join(args.test_dir, 'noisy/')
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector

    target_dir = args.enhanced_dir
    ensure_dir(target_dir)

    # Settings
    sr = 16000
    snr = args.snr
    N = args.N
    corrector_steps = args.corrector_steps

    # Create list of tasks
    task_list = []
    gpu_id = 0
    for noisy_file in sorted(glob.glob('{}/*.wav'.format(noisy_dir))):
        task = {
            'noisy_file': noisy_file,
            'target_dir': target_dir,
            'checkpoint_file': checkpoint_file,
            'corrector_cls': corrector_cls,
            'snr': snr,
            'N': N,
            'corrector_steps': corrector_steps,
            'gpu_id': gpu_id
        }
        task_list.append(task)
        
        gpu_id = (gpu_id + 1) % 8


    # Start processes
    print('Starting {} tasks'.format(len(task_list)))
    with mp.get_context("spawn").Pool(8) as pool:
        results = [pool.apply_async(process_file, args=(task,)) for task in task_list]
        output = [p.get() for p in results]
