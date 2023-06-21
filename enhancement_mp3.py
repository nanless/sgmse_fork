import glob
from argparse import ArgumentParser
from os.path import join

import torch
import torch.multiprocessing as mp
from soundfile import write
from torchaudio import load
from tqdm import tqdm

from sgmse.model import ScoreModel
from sgmse.util.other import ensure_dir, pad_spec


def process_file(noisy_file, target_dir, checkpoint_file, corrector_cls, snr, N, corrector_steps):
    # Load score model
    model = ScoreModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()

    filename = noisy_file.split('/')[-1]

    # Load wav
    y, _ = load(noisy_file)
    T_orig = y.size(1)

    # Normalize
    norm_factor = y.abs().max()
    y = y / norm_factor

    # Prepare DNN input
    Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
    Y = pad_spec(Y)

    # Reverse sampling
    sampler = model.get_pc_sampler(
        'reverse_diffusion', corrector_cls, Y.cuda(), N=N,
        corrector_steps=corrector_steps, snr=snr)
    sample, _ = sampler()

    # Backward transform in time domain
    x_hat = model.to_audio(sample.squeeze(), T_orig)

    # Renormalize
    x_hat = x_hat * norm_factor

    # Write enhanced wav file
    write(join(target_dir, filename), x_hat.cpu().numpy(), 16000)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data (must have subdirectory noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--extra_dir", type=str, default=None, help="Extra directory")
    args = parser.parse_args()

    noisy_dir = join(args.test_dir, 'noisy/')
    extra_dir = args.extra_dir
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector

    target_dir = args.enhanced_dir
    ensure_dir(target_dir)

    # Settings
    sr = 16000
    snr = args.snr
    N = args.N
    corrector_steps = args.corrector_steps

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
    real_noisy_files = []

    if extra_dir is not None:
        extra_files = sorted(glob.glob('{}/*.wav'.format(extra_dir)))
        extra_file_names = [extra_file.split('/')[-1] for extra_file in extra_files]
        for noisy_file in noisy_files:
            noisy_file_name = noisy_file.split('/')[-1]
            if noisy_file_name not in extra_file_names:
                real_noisy_files.append(noisy_file)

    print("noisy files: ", len(noisy_files))
    print("real noisy files: ", len(real_noisy_files))

    with mp.get_context("spawn").Pool(2) as pool:
        results = [pool.apply_async(process_file, args=(noisy_file, target_dir, checkpoint_file, corrector_cls, snr, N, corrector_steps)) for noisy_file in real_noisy_files]
        output = [p.get() for p in results]
