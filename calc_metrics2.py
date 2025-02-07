from os.path import join 
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
from pystoi import stoi
from sgmse.util.other import energy_ratios, mean_std
from multiprocessing import Pool

def evaluate_metric(noisy_file, clean_dir, enhanced_dir, sr):
    filename = noisy_file.split('/')[-1]
    x, _ = read(join(clean_dir, filename))
    y, _ = read(noisy_file)
    n = y - x 
    x_method, _ = read(join(enhanced_dir, filename))
    result = {
        "filename": filename,
        "pesq": pesq(sr, x, x_method, 'wb'),
        "estoi": stoi(x, x_method, sr, extended=True),
        "si_sdr": energy_ratios(x_method, x, n)[0],
        "si_sir": energy_ratios(x_method, x, n)[1],
        "si_sar": energy_ratios(x_method, x, n)[2]        
    }
    return result

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the original test data (must have subdirectories clean/ and noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    args = parser.parse_args()

    test_dir = args.test_dir
    clean_dir = join(test_dir, "clean/")
    noisy_dir = join(test_dir, "noisy/")
    enhanced_dir = args.enhanced_dir

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": []}
    sr = 16000

    # Evaluate standard metrics
    pool = Pool()
    noisy_files = sorted(glob('{}/*.wav'.format(noisy_dir)))
    results = []
    for result in tqdm(pool.imap_unordered(lambda x: evaluate_metric(x, clean_dir, enhanced_dir, sr), noisy_files)):
        results.append(result)
    pool.close()
    pool.join()

    # Save results as DataFrame    
    df = pd.DataFrame(results)

    # POLQA evaluation  -  requires POLQA license and server, uncomment at your own peril.
    # This is batch processed for speed reasons and thus runs outside the for loop.
    # if not basic:
    #     clean_files = sorted(glob('{}/*.wav'.format(clean_dir)))
    #     enhanced_files = sorted(glob('{}/*.wav'.format(enhanced_dir)))
    #     clean_audios = [read(clean_file)[0] for clean_file in clean_files]
    #     enhanced_audios = [read(enhanced_file)[0] for enhanced_file in enhanced_files]
    #     polqa_vals = polqa(clean_audios, enhanced_audios, 16000, save_to=None)
    #     polqa_vals = [val[1] for val in polqa_vals]
    #     # Add POLQA column to DataFrame
    #     df['polqa'] = polqa_vals

    # Print results
    print(enhanced_dir)
    #print("POLQA: {:.2f} ± {:.2f}".format(*mean_std(df["polqa"].to_numpy())))
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))

    # Save DataFrame as csv file
    df.to_csv(join(enhanced_dir, "_results.csv"), index=False)
