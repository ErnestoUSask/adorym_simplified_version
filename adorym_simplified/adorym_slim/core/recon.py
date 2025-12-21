"""
Simplified reconstruction driver mirroring the original 2D ptychography demo.
"""

from __future__ import annotations

import datetime
import importlib.util
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

_dxchange_spec = importlib.util.find_spec("dxchange")
if _dxchange_spec:
    import dxchange  # type: ignore
else:  # pragma: no cover - optional dependency
    dxchange = None  # type: ignore

from ..backends import Array, ensure_complex, random_complex, to_magnitude, zeros
from ..io.data_loaders import DiffractionDataset, load_dataset, load_initial_guess
from ..utils import ensure_dir
from .optim import AdamOptimizer
from .ptycho_forward import aperture_probe, initialize_object, loss_and_gradients


@dataclass
class ReconstructionResult:
    object: Array
    probes: List[Array]
    loss_history: List[float]


def _setup_output_folder(save_path: str, output_folder: str | None) -> str:
    if output_folder is None:
        timestr = str(datetime.datetime.today())
        timestr = timestr[: timestr.find(".")]
        for char in [":", "-", " "]:
            replacement = "_" if char == " " else ""
            timestr = timestr.replace(char, replacement)
        output_folder = f"recon_{timestr}"
    if save_path != "." and not os.path.isabs(output_folder):
        output_folder = os.path.join(save_path, output_folder)
    ensure_dir(output_folder)
    return output_folder


def reconstruct_ptychography(
    fname: str,
    obj_size: Tuple[int, int, int],
    probe_pos: Sequence[Sequence[float]] | None = None,
    theta_st: float = 0,
    theta_end: float = 0,
    n_theta: int | None = None,
    theta_downsample: int | None = None,
    energy_ev: float | None = None,
    psize_cm: float | None = None,
    free_prop_cm: float | str | None = None,
    raw_data_type: str = "magnitude",
    is_minus_logged: bool = False,
    slice_pos_cm_ls: Sequence[float] | None = None,
    n_epochs: int = 1000,
    crit_conv_rate: float = 0.03,
    max_nepochs: int = 200,
    regularizers=None,
    alpha_d=None,
    alpha_b=None,
    gamma: float = 1e-6,
    minibatch_size: int | None = None,
    multiscale_level: int = 1,
    n_epoch_final_pass=None,
    initial_guess=None,
    random_guess_means_sigmas=(1.0, 0.0, 0.001, 0.002),
    n_batch_per_update: int = 1,
    reweighted_l1: bool = False,
    interpolation: str = "bilinear",
    update_scheme: str = "immediate",
    unknown_type: str = "delta_beta",
    randomize_probe_pos: bool = False,
    common_probe_pos: bool = True,
    fix_object: bool = False,
    optimize_object: bool = True,
    optimizer: AdamOptimizer | str = "adam",
    learning_rate: float = 1e-5,
    update_using_external_algorithm=None,
    optimizer_batch_number_increment: str = "angle",
    finite_support_mask_path=None,
    shrink_cycle=None,
    shrink_threshold: float = 1e-9,
    object_type: str = "normal",
    non_negativity: bool = False,
    forward_model: str = "auto",
    forward_algorithm: str = "fresnel",
    ctf_lg_kappa: float = 1.7,
    binning: int = 1,
    fresnel_approx: bool = True,
    pure_projection: bool = False,
    two_d_mode: bool = False,
    probe_type: str = "gaussian",
    probe_initial=None,
    probe_extra_defocus_cm=None,
    n_probe_modes: int = 1,
    shared_probe_among_angles: bool = True,
    rescale_probe_intensity: bool = False,
    loss_function_type: str = "lsq",
    poisson_multiplier: float = 1.0,
    beamstop=None,
    normalize_fft: bool = False,
    safe_zone_width: int = 0,
    scale_ri_by_k: bool = True,
    sign_convention: int = 1,
    fourier_disparity: bool = False,
    save_path: str = ".",
    output_folder: str | None = None,
    save_intermediate: bool = False,
    save_intermediate_level: str = "batch",
    save_history: bool = True,
    store_checkpoint: bool = True,
    use_checkpoint: bool = True,
    force_to_use_checkpoint: bool = False,
    n_batch_per_checkpoint: int = 10,
    full_intermediate: bool = True,
    save_stdout: bool = True,
    cpu_only: bool = False,
    core_parallelization: bool = True,
    gpu_index: int = 0,
    n_dp_batch: int = 20,
    distribution_mode=None,
    dist_mode_n_batch_per_update=None,
    precalculate_rotation_coords: bool = True,
    cache_dtype: str = "float32",
    rotate_out_of_loop: bool = False,
    n_split_mpi_ata="auto",
    optimize_probe: bool = False,
    probe_learning_rate: float = 1e-5,
    optimizer_probe: AdamOptimizer | None = None,
    probe_update_delay: int = 0,
    probe_update_limit=None,
    optimize_probe_defocusing: bool = False,
    probe_defocusing_learning_rate: float = 1e-5,
    optimizer_probe_defocusing=None,
    optimize_probe_pos_offset: bool = False,
    probe_pos_offset_learning_rate: float = 1e-2,
    optimizer_probe_pos_offset=None,
    optimize_prj_pos_offset: bool = False,
    prj_pos_offset_learning_rate: float = 1e-2,
    optimizer_prj_pos_offset=None,
    optimize_all_probe_pos: bool = False,
    all_probe_pos_learning_rate: float = 1e-2,
    optimizer_all_probe_pos: AdamOptimizer | None = None,
    optimize_slice_pos: bool = False,
    slice_pos_learning_rate: float = 1e-4,
    optimizer_slice_pos=None,
    optimize_free_prop: bool = False,
    free_prop_learning_rate: float = 1e-2,
    optimizer_free_prop=None,
    optimize_prj_affine: bool = False,
    prj_affine_learning_rate: float = 1e-3,
    optimizer_prj_affine=None,
    optimize_tilt: bool = False,
    tilt_learning_rate: float = 1e-3,
    initial_tilt=None,
    optimize_ctf_lg_kappa: bool = False,
    ctf_lg_kappa_learning_rate: float = 1e-3,
    optimizer_ctf_lg_kappa=None,
    other_params_update_delay: int = 0,
    use_epie: bool = False,
    epie_alpha: float = 0.8,
    dynamic_rate: bool = True,
    pupil_function=None,
    probe_circ_mask: float = 0.9,
    dynamic_dropping: bool = False,
    dropping_threshold: float = 8e-5,
    backend: str = "autograd",
    debug: bool = False,
    t_max_min=None,
    xpu: bool = False,
    run_bfloat16: bool = False,
    run_float64: bool = False,
    full_intermediate: bool = True,
    **kwargs,
) -> Dict[str, object]:
    """
    Execute a pared-down reconstruction loop compatible with the original demo parameters.
    """

    dataset_path = os.path.join(save_path, fname) if not os.path.isabs(fname) else fname
    dataset: DiffractionDataset = load_dataset(dataset_path, probe_pos_override=probe_pos)
    output_folder = _setup_output_folder(save_path, output_folder)

    init_guess_processed = initial_guess
    if isinstance(initial_guess, str):
        # Allow callers to pass "checkpoint" to resume from previous epoch.
        try:
            epoch_num = int(initial_guess)
            init_guess_processed = load_initial_guess(save_path, output_folder, epoch_num)
        except (ValueError, RuntimeError):
            init_guess_processed = None
    object_slice = initialize_object(obj_size, init_guess_processed, random_guess_means_sigmas)

    wavelength_cm = None
    if energy_ev:
        h_planck_ev_s = 4.135667696e-15
        speed_of_light_cm_s = 2.99792458e10
        wavelength_cm = (h_planck_ev_s * speed_of_light_cm_s) / energy_ev

    if probe_type == "aperture_defocus":
        probes = aperture_probe(
            radius=kwargs.get("aperture_radius", 10),
            det_shape=dataset.detector_shape,
            n_modes=n_probe_modes,
            beamstop_radius=kwargs.get("beamstop_radius", 0),
            defocus_cm=kwargs.get("probe_defocus_cm", probe_extra_defocus_cm),
            wavelength_cm=wavelength_cm,
            psize_cm=psize_cm,
        )
    else:
        probes = [random_complex(dataset.detector_shape, mean=1.0, std=1e-3) for _ in range(n_probe_modes)]

    obj_opt = optimizer if isinstance(optimizer, AdamOptimizer) else AdamOptimizer("obj", output_folder, distribution_mode, {"step_size": learning_rate})
    probe_opt = optimizer_probe or AdamOptimizer("probe", output_folder, distribution_mode, {"step_size": probe_learning_rate})

    loss_history: List[float] = []
    positions = list(dataset.probe_positions)
    if len(positions) < len(dataset.intensities):
        fallback_pos = positions[-1] if positions else (dataset.detector_shape[0] / 2.0, dataset.detector_shape[1] / 2.0)
        positions.extend([fallback_pos] * (len(dataset.intensities) - len(positions)))
    positions = positions[: len(dataset.intensities)]
    if randomize_probe_pos:
        rng = np.random.default_rng(0)
        rng.shuffle(positions)

    for epoch in range(n_epochs):
        epoch_losses: List[float] = []
        for idx, (intensity, pos) in enumerate(zip(dataset.intensities, positions)):
            loss, obj_grad, probe_grads, top_left = loss_and_gradients(
                object_slice, probes, intensity, position=pos, normalize_fft=normalize_fft
            )
            if optimize_object and not fix_object:
                object_slice = obj_opt.step("obj", object_slice, obj_grad)
            if optimize_probe:
                updated_probes: List[Array] = []
                for mode_idx, (probe, grad) in enumerate(zip(probes, probe_grads)):
                    updated_probes.append(probe_opt.step(f"probe_{mode_idx}", probe, grad))
                probes = updated_probes
            epoch_losses.append(loss)
            if save_intermediate:
                _save_epoch_outputs(output_folder, epoch, object_slice, probes, full_intermediate)
        loss_history.extend(epoch_losses)

    if save_intermediate:
        _save_epoch_outputs(output_folder, "final", object_slice, probes, full_intermediate)

    return {
        "object": object_slice,
        "probes": probes,
        "loss_history": loss_history,
        "output_folder": output_folder,
    }


def _save_epoch_outputs(folder: str, epoch, obj: Array, probes: List[Array], full: bool) -> None:
    ensure_dir(folder)
    tag = f"epoch_{epoch}"
    obj_path = os.path.join(folder, f"{tag}_object.npy")
    np.save(obj_path, obj)
    if dxchange is not None:
        epoch_dir = os.path.join(folder, f"epoch_{epoch}")
        ensure_dir(epoch_dir)
        dxchange.write_tiff(obj.real, os.path.join(epoch_dir, "delta_ds_1.tiff"), overwrite=True)  # type: ignore
        dxchange.write_tiff(obj.imag, os.path.join(epoch_dir, "beta_ds_1.tiff"), overwrite=True)  # type: ignore
    for idx, probe in enumerate(probes):
        probe_path = os.path.join(folder, f"{tag}_probe_mode_{idx}.npy")
        np.save(probe_path, probe if full else probe.real)
