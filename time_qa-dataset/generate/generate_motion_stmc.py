#!/usr/bin/env python3

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from json import loads
from pathlib import Path
from typing import Annotated, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import typer
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rtpt import RTPT

stmc_path = Path(__file__).absolute().parents[2] / "stmc"
sys.path.append(str(stmc_path))

from app_config import CONFIG, FPS
from src.config import read_config
from src.model.text_encoder import TextToEmb
from src.stmc import BODY_PARTS_LST, TextInterval, process_timelines
from src.tools.extract_joints import extract_joints
from src.tools.smpl_layer import SMPLH


def check_dataset_path(ctx: typer.Context, dataset_path: Path):
    if not dataset_path.exists():
        raise typer.BadParameter(f"Dataset path {dataset_path} does not exist.")

    if not dataset_path.glob("*/*.json"):
        raise RuntimeError(f"No JSON files found in {dataset_path}/*/")

    return dataset_path.absolute()


class SMPLHVariant(str, Enum):
    neutral = "neutral"
    male = "male"
    female = "female"


def generate_motion(
    dataset_path: Path = typer.Option(
        help=(
            "Path to the text prompts dataset. "
            "Expected to contain folders with integer names and 'data.json' files in each folder. "
            "These folders are where the results will be stored."
        ),
        callback=check_dataset_path,
    ),
    only_data_id: Optional[int] = None,
    gpu_id: int = typer.Option(0, help="If a GPU is available, use this GPU for the diffusion model."),
    seed: int = typer.Option(
        0,
        help="Seed to use for the random number generator. A seed of -1 means to use the default seed.",
    ),
    smplh_variant: Annotated[
        SMPLHVariant, typer.Option(case_sensitive=False, help="The style of the SMPL-H model to use.")
    ] = SMPLHVariant.neutral,
    batch_size: int = typer.Option(
        256,
        help="How many instances to process in parallel in the diffusion model.",
    ),
):
    # Share access to the console between all threads
    console = Console()

    all_instance_files = list(dataset_path.glob("*/*.json"))
    all_instance_files.sort()

    if only_data_id is not None:
        all_instance_files = [all_instance_files[only_data_id]]

    rtpt = RTPT(
        name_initials="FD/MK",
        experiment_name="tsqa-motion-generation",
        max_iterations=len(all_instance_files),
    )
    rtpt.start()

    missing_instances = [
        instance_path
        for instance_path in all_instance_files
        if not (instance_path.parent / "data.hdf5").exists()
    ]
    for _ in range(len(all_instance_files) - len(missing_instances)):
        rtpt.step()

    console.print(
        f"Missing {len(missing_instances)} instances ({len(missing_instances) * 100 / len(all_instance_files):.2f}%)"
    )

    console.print("Loading diffusion model ...", end="")
    motion_maker = MotionMaker(gpu_id=gpu_id, smplh_variant=smplh_variant)
    console.print(" done")

    # make a simple progress bar with rich based on len(missing_instances)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn("[progress.percentage]({task.percentage:>3.0f}%)"),
        TimeRemainingColumn(),
        expand=True,
        speed_estimate_period=20 * 60,  # Very slow estimate, since our tasks are very long
        console=console,
    ) as progress:
        task = progress.add_task("Generating motion sequences", total=len(missing_instances))

        for batch_id, instance_paths in enumerate(batched(missing_instances, batch_size)):
            prompts = []
            for instance_path in instance_paths:
                data = loads(instance_path.read_text(encoding="utf-8"))

                prompts.append(
                    [
                        TextInterval(
                            text=f"a person is {prompt['action']}",
                            start=int(round(prompt["start"] * FPS)),
                            end=int(round(prompt["end"] * FPS)),
                            bodyparts=BODY_PARTS_LST,  # All body parts (TODO)
                        )
                        for prompt in data["prompt_sequence"]
                    ]
                )

            data_results = motion_maker.generate(
                prompts=prompts,
                seed=seed + batch_id,  # Different seed for each batch
                guidance=3.0,
                overlap_s=1.0,
                progress=None,  # partial(track, console=console, description="Diffusing motion"),
            )

            for i, instance_path in enumerate(instance_paths):
                with h5py.File(instance_path.parent / "data.hdf5", "w") as hdf5_file:
                    hdf5_file.create_dataset(
                        "smplrifke", data=data_results.rifke_feats[i], compression="gzip"
                    )
                    hdf5_file.create_dataset("joints", data=data_results.joints[i], compression="gzip")
                    hdf5_file.create_dataset("vertices", data=data_results.vertices[i], compression="gzip")
                    smpl_group = hdf5_file.create_group("smpl")
                    for key, value in data_results.smpl[i].items():
                        match key:
                            case "mocap_framerate":
                                assert value == 20  # We cannot easily store this in the HDF5 file
                            case "joints":
                                pass  # This is redundant
                            case _:
                                smpl_group.create_dataset(key, data=value, compression="gzip")

                rtpt.step()
                progress.update(task, advance=1)


@dataclass
class BatchedDataResult:
    rifke_feats: Sequence[np.ndarray]
    joints: Sequence[np.ndarray]
    vertices: Sequence[np.ndarray]
    smpl: Sequence[dict[str, torch.Tensor | int]]


class MotionMaker:
    # We want to encapsulate the motion generation in a class, such that
    # the model is only loaded once and when needed

    def __init__(self, gpu_id: int = 0, smplh_variant: SMPLHVariant = SMPLHVariant.neutral) -> None:
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = f"cuda:{gpu_id}"

        config = CONFIG.copy()
        c = OmegaConf.create(config)
        c.run_dir = str(stmc_path / "pretrained_models" / "mdm-smpl_clip_smplrifke_humanml3d")
        cfg = read_config(c.run_dir)
        self.motion_features = cfg.motion_features

        ckpt_name = c.ckpt
        ckpt_path = os.path.join(c.run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        # Diffusion model
        # update the folder first, in case it has been moved
        cfg.diffusion.motion_normalizer.base_dir = os.path.join(c.run_dir, "motion_stats")
        cfg.diffusion.text_normalizer.base_dir = os.path.join(c.run_dir, "text_stats")
        self.diffusion = instantiate(cfg.diffusion)
        self.diffusion.load_state_dict(ckpt["state_dict"])
        # Evaluation mode
        self.diffusion.eval()
        self.diffusion.to(self.device)

        modelpath = cfg.data.text_encoder.modelname
        mean_pooling = cfg.data.text_encoder.mean_pooling
        self.text_model = TextToEmb(modelpath=modelpath, mean_pooling=mean_pooling, device=self.device)

        self.smplh = SMPLH(
            path=str(stmc_path / "deps" / "smplh"),
            jointstype="both",
            input_pose_rep="axisangle",
            gender=smplh_variant,
        )

    def generate(
        self,
        prompts: Sequence[Sequence[TextInterval]],
        seed: int,
        guidance: float,
        overlap_s: float,  # The overlap between the actions in seconds
        progress: Callable,
    ) -> BatchedDataResult:
        interval_overlap = int(FPS * overlap_s)

        # process the timelines
        infos = process_timelines(prompts, interval_overlap)
        infos["output_lengths"] = infos["max_t"]
        infos["featsname"] = self.motion_features
        infos["guidance_weight"] = guidance

        if seed != -1:
            pl.seed_everything(seed)

        with torch.no_grad():
            tx_emb = self.text_model(infos["all_texts"])
            tx_emb_uncond = self.text_model(["" for _ in infos["all_texts"]])

            if isinstance(tx_emb, torch.Tensor):
                tx_emb = {
                    "x": tx_emb[:, None],
                    "length": torch.tensor([1 for _ in range(len(tx_emb))]).to(self.device),
                }
                tx_emb_uncond = {
                    "x": tx_emb_uncond[:, None],
                    "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(self.device),
                }

            # The actual diffusion process
            data_smpl_rifke_feats: torch.Tensor = self.diffusion(
                tx_emb, tx_emb_uncond, infos, progress_bar=progress
            ).cpu()

            # Extract joint data
            data_joints = [
                extract_joints(
                    data[:output_length, ...],
                    self.motion_features,
                    fps=FPS,
                    value_from="joints",
                    smpl_layer=self.smplh,  # Not really needed
                )["joints"]
                for data, output_length in zip(data_smpl_rifke_feats, infos["output_lengths"])
            ]
            data_converted = [
                extract_joints(
                    data[:output_length, ...],
                    self.motion_features,
                    fps=FPS,
                    value_from="smpl",
                    smpl_layer=self.smplh,
                )
                for data, output_length in zip(data_smpl_rifke_feats, infos["output_lengths"])
            ]
            # We deliberately don't use data_converted[0]["joints"], since neither does the original app.py

            return BatchedDataResult(
                rifke_feats=data_smpl_rifke_feats,
                joints=data_joints,
                vertices=[d["vertices"] for d in data_converted],
                smpl=[d["smpldata"] for d in data_converted],
            )


def batched(iterable, n):
    """batched('ABCDEFG', 3) → ABC DEF G"""

    # Copied from Pytorch 3.12 (https://docs.python.org/3/library/itertools.html#itertools.batched)
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch
