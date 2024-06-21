#!/usr/bin/env python3

import os
import sys
from collections.abc import Callable
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from shutil import rmtree
from typing import Optional

import h5py
import typer
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import track

stmc_path = Path(__file__).absolute().parents[2] / "stmc"
sys.path.append(str(stmc_path))

from app_config import CONFIG


def check_dataset_path(ctx: typer.Context, dataset_path: Path):
    if not dataset_path.exists():
        raise typer.BadParameter(f"Dataset path {dataset_path} does not exist.")

    if not dataset_path.glob("*/*.json"):
        raise RuntimeError(f"No JSON files found in {dataset_path}/*/")

    return dataset_path.absolute()


def animate_motion(
    dataset_path: Path = typer.Option(
        help=(
            "Path to the text prompts dataset. "
            "Expected to contain folders with integer names and 'data.json' files in each folder. "
            "These folders are where the results will be stored."
        ),
        callback=check_dataset_path,
    ),
    only_data_id: Optional[int] = None,
    n_parallel: int = typer.Option(
        1, help="Use carefully. The process is CPU hungry and already parallelizes some rendering."
    ),
    render_simple: bool = True,
    render_smpl: bool = True,
):
    # Share access to the console between all threads
    console = Console()

    all_instance_files = list(dataset_path.glob("*/*.json"))
    all_instance_files.sort()

    if only_data_id is not None:
        all_instance_files = [all_instance_files[only_data_id]]

    missing_instances = [
        instance_path
        for instance_path in all_instance_files
        if (instance_path.parent / "data.hdf5").exists()
        and (
            (render_simple and not (instance_path.parent / "render_simple.mp4").exists())
            or (render_smpl and not (instance_path.parent / "render_smpl.mp4").exists())
        )
    ]

    console.print(
        f"Missing {len(missing_instances)} instances ({len(missing_instances) * 100 / len(all_instance_files):.2f}%)"
    )

    if not missing_instances:
        console.print("Nothing to render")
        return

    console.print("Loading animation engine ...", end="")
    renderer = MotionRenderer(render_simple=render_simple, render_smpl=render_smpl)
    console.print(" done")

    def process_single(instance_path: Path) -> None:
        with h5py.File(instance_path.parent / "data.hdf5", "r") as hdf5_file:
            data_joints = hdf5_file["joints"][:]
            data_vertices = hdf5_file["vertices"][:]

        renderer.render(
            data_joints=data_joints,
            data_vertices=data_vertices,
            directory=instance_path.parent,
            progress=partial(track, console=console, description="Rendering SMPL"),
        )

    with ThreadPool(processes=n_parallel) as pool:
        for _ in track(
            pool.imap_unordered(process_single, missing_instances),
            total=len(missing_instances),
            console=console,
        ):
            pass


class MotionRenderer:
    # We want to encapsulate the motion generation in a class, such that
    # the model is only loaded once and when needed

    def __init__(self, render_simple: bool = True, render_smpl: bool = True):
        c = OmegaConf.create(CONFIG.copy())
        c.run_dir = str(stmc_path / "pretrained_models" / "mdm-smpl_clip_smplrifke_humanml3d")

        self.joints_renderer = instantiate(c.joints_renderer)
        self.smpl_renderer = instantiate(c.smpl_renderer)

        self.render_simple = render_simple
        self.render_smpl = render_smpl

        # Make sure to use the right offscreen renderer
        # See: https://pyrender.readthedocs.io/en/latest/examples/offscreen.html
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    def render(self, data_joints, data_vertices, directory: Path, progress: Callable) -> None:
        if self.render_simple:
            self.joints_renderer(
                data_joints, title="", output=directory / "render_simple.mp4", canonicalize=False
            )
        if self.render_smpl:
            self.smpl_renderer(data_vertices, output=directory / "render_smpl.mp4", progress_bar=None)
            # Clean up the temporary directory
            rmtree(directory / "render_smpl")
