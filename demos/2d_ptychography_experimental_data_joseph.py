from pathlib import Path

import h5py
import tifffile

from adorym.ptychography import reconstruct_ptychography
import adorym
import numpy as np
import dxchange
import datetime
import argparse
import os

timestr = str(datetime.datetime.today())
timestr = timestr[:timestr.find('.')]
for i in [':', '-', ' ']:
    if i == ' ':
        timestr = timestr.replace(i, '_')
    else:
        timestr = timestr.replace(i, '')

from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default='None')
parser.add_argument('--save_path', default='cone_256_foam_ptycho')
parser.add_argument('--output_folder', default='test')  # Will create epoch folders under this
args = parser.parse_args()
epoch = args.epoch
if epoch == 'None':
    epoch = 0
    init = None
else:
    epoch = int(epoch)
    if epoch == 0:
        init = None
    else:
        init_delta = dxchange.read_tiff(
            os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init_beta = dxchange.read_tiff(
            os.path.join(args.save_path, args.output_folder, 'epoch_{}/beta_ds_1.tiff'.format(epoch - 1)))
        print(os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init = [np.array(init_delta[...]), np.array(init_beta[...])]

output_folder = r'D:\Joseph Reconstruction\~ Reconstructions\250926_1225'
distribution_mode = None
optimizer_obj = adorym.AdamOptimizer('obj', output_folder=output_folder, distribution_mode=distribution_mode,
                                     options_dict={'step_size': 1e-3})
optimizer_probe = adorym.AdamOptimizer('probe', output_folder=output_folder, distribution_mode=distribution_mode,
                                       options_dict={'step_size': 1e-3, 'eps': 1e-7})
optimizer_all_probe_pos = adorym.AdamOptimizer('probe_pos_correction', output_folder=output_folder,
                                               distribution_mode=distribution_mode,
                                               options_dict={'step_size': 1e-2})

h5_files_folder_path = Path(
    r'C:\Users\erobe\OneDrive - University of Saskatchewan\Resources\Data\Joseph - PtychoRec\~ Original Data - Nano Spheres System\Extracted Data - H5 - 571 eV\Cropping Size - 1920')
h5_files = list(h5_files_folder_path.glob("*.h5"))

bg_centered = tifffile.imread(r"C:\Users\erobe\OneDrive - University of Saskatchewan\Resources\Data\Joseph - PtychoRec\Reconstruction Data\background_centered_1920.tiff")

mag0 = tifffile.imread(
    r"C:\Users\erobe\OneDrive - University of Saskatchewan\Resources\Data\Joseph - PtychoRec\Reconstruction Data\probe_abs.tif")
ph0 = tifffile.imread(
    r"C:\Users\erobe\OneDrive - University of Saskatchewan\Resources\Data\Joseph - PtychoRec\Reconstruction Data\probe_angle.tif")

probe_mag_phase = [mag0, ph0]
probe_mag_phase = np.array(probe_mag_phase)

for h5_path in h5_files:
    with h5py.File(h5_path, "r") as f:
        rec_image_size = f["metadata/reconstruction_size"][()]  # returns a numpy array
        rec_y, rec_x = map(int, rec_image_size)  # ensure plain ints

    # Create timestamp in yymmdd_hhmm format
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    # Base name of the h5 file (without extension)
    base_name = h5_path.stem

    # Construct output folder path:
    #  main_folder / f"{timestamp}_{base_name}"
    main_folder = r'D:\Joseph Reconstruction\~ Reconstructions'
    output_folder = f"{timestamp}_{base_name}"

    params_2idd_gpu = {'fname': h5_path,
                       'theta_st': 0,
                       'theta_end': 0,
                       'n_epochs': 400,
                       'obj_size': (rec_y, rec_x, 1),
                       'two_d_mode': True,
                       # 'background_data': r"C:\Users\erobe\OneDrive - University of Saskatchewan\Resources\Data\Joseph - PtychoRec\Reconstruction Data\background_centered_1920.tiff",
                       'background_data': r"C:\Users\erobe\OneDrive - University of Saskatchewan\Resources\Data\Joseph - PtychoRec\Reconstruction Data\background_centered_1920.tiff",
                       'use_coherent_master_slave': False,
                       # 'energy_ev': 571,
                       # 'psize_cm': 1.3365e-06,
                       'minibatch_size': 10,
                       'output_folder': output_folder,
                       'cpu_only': False,
                       'save_path': main_folder,
                       'use_checkpoint': False,
                       'n_epoch_final_pass': None,
                       'save_intermediate': False,
                       'full_intermediate': True,
                       'initial_guess': None,
                       'random_guess_means_sigmas': (1., 0., 0.001, 0.002),
                       'n_dp_batch': 350,
                       # ===============================plane
                       'n_probe_modes': 1,
                       # 'probe_type': 'aperture_defocus',
                       # 'aperture_radius': 1,
                       # 'beamstop_radius': 25,
                       # 'probe_defocus_cm': 0.0046,
                       'probe_type': 'supplied',
                       'probe_initial': probe_mag_phase,
                       # ===============================
                       'rescale_probe_intensity': True,
                       'free_prop_cm': 'inf',
                       'backend': 'pytorch',
                       'raw_data_type': 'intensity',
                       'beamstop': None,
                       'randomize_probe_pos': False,
                       'optimizer': optimizer_obj,
                       'optimize_probe': True,
                       'optimizer_probe': optimizer_probe,
                       'optimize_all_probe_pos': True,
                       'optimizer_all_probe_pos': optimizer_all_probe_pos,
                       'save_history': True,
                       'update_scheme': 'immediate',
                       'unknown_type': 'real_imag',
                       'save_stdout': True,
                       'loss_function_type': 'poisson',
                       }

    params = params_2idd_gpu

    # Ensure the output directory exists
    output_dir = Path(main_folder) / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Path of the parameter text file
    param_file = output_dir / "reconstruction_params.txt"

    # Save all parameters in a human-readable format
    with open(param_file, "w") as ftxt:
        for k, v in params.items():
            ftxt.write(f"{k}: {v}\n")

    print(f"Parameters saved to: {param_file}")

    reconstruct_ptychography(**params)
