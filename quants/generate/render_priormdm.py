import json
from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import typer
from datasets import VerificationMode, load_dataset
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rich.console import Console
from rich.table import Table
from tables import open_file

t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
HML_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def plot_3d_motion(
    path: str,
    kinematic_tree: np.ndarray,
    data: np.ndarray,
    normalize: bool = True,
    painting_features: list[str] = [],
    labels: bool = False,
    label_vector: bool = False,
    display_vector: bool = False,
):
    fps = 30

    if normalize:
        # Normalize data relative to the initial position of the root joint
        height_offset = data[:, :, 1].min()
        data[:, :, 1] -= height_offset
        # This code calculates the minimum value of the y-coordinate across all
        # data points and subtracts it from all y-coordinates.
        # This effectively shifts the animation so that the lowest point (in terms of y-coordinate) is at y = 0.
        # This is useful if your data includes negative y-values or if you want the motion to appear as if it's starting from ground level.

        data[..., 0] -= data[:, 0:1, 0]  # Normalize x relative to the first joint
        data[..., 2] -= data[:, 0:1, 2]  # Normalize z relative to the first joint
        # used to normalize the x and z
        # coordinates relative to the initial position of the root joint in each frame.
        # This ensures that the motion is visualized relative to the starting position of this joint,
        # effectively centering the animation around the root joint’s motion in the horizontal plane.

    # Determine data ranges for better axes limits
    max_range = np.array([data[:, :, i].max() - data[:, :, i].min() for i in range(3)]).max() / 2.0
    mid_x = (data[:, :, 0].max() + data[:, :, 0].min()) * 0.5
    mid_z = (data[:, :, 2].max() + data[:, :, 2].min()) * 0.5

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))

    def init_plot():
        # Dynamically set axes limits
        ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
        ax.set_ylim3d([0, max_range * 2])  # Adjust this to ensure the floor is visible
        ax.set_zlim3d([mid_z - max_range, mid_z + max_range])
        ax.grid(False)
        ax.axis("off")

        # Floor plane
        floor_size = max_range * 2  # Ensures the floor is a square and adequately large
        floor_verts = np.array(
            [
                [mid_x - floor_size, 0, mid_z - floor_size],
                [mid_x - floor_size, 0, mid_z + floor_size],
                [mid_x + floor_size, 0, mid_z + floor_size],
                [mid_x + floor_size, 0, mid_z - floor_size],
            ]
        )
        floor = Poly3DCollection([floor_verts], alpha=0.3, facecolor="grey")
        ax.add_collection3d(floor)

    def update(frame):
        ax.clear()
        init_plot()
        ax.view_init(elev=100, azim=-90)  # Adjust view angles if necessary

        # Dynamic plot adjustments based on frame data
        for chain in kinematic_tree:
            ax.plot3D(
                data[frame, chain, 0],
                data[frame, chain, 1],
                data[frame, chain, 2],
                linewidth=2.0,
                color="orange",
            )

        # Plot painting features with trajectories and text labels
        for feature in painting_features:
            if feature in HML_JOINT_NAMES:
                feat_index = HML_JOINT_NAMES.index(feature)
                # Plot trajectory
                ax.plot3D(
                    data[: frame + 1, feat_index, 0],
                    data[: frame + 1, feat_index, 1],
                    data[: frame + 1, feat_index, 2],
                    linewidth=2.0,
                    color="blue",
                )
                # Add text label with coordinates
                coord_text = None
                if labels:
                    coord_text = feature

                if label_vector:
                    coord_text = f"{feature} ({data[frame, feat_index, 0]:.2f}, {data[frame, feat_index, 1]:.2f}, {data[frame, feat_index, 2]:.2f})"

                if display_vector:
                    # Display vector from the root joint to the feature
                    ax.quiver(
                        data[frame, 0, 0],
                        data[frame, 0, 1],
                        data[frame, 0, 2],
                        data[frame, feat_index, 0] - data[frame, 0, 0],
                        data[frame, feat_index, 1] - data[frame, 0, 1],
                        data[frame, feat_index, 2] - data[frame, 0, 2],
                        color="green",
                        arrow_length_ratio=0.1,
                        linestyle="dotted",
                    )
                # coord_text = f"{feature} ({data[frame, feat_index, 0]:.2f}, {data[frame, feat_index, 1]:.2f}, {data[frame, feat_index, 2]:.2f})"
                if coord_text is not None:
                    ax.text(
                        data[frame, feat_index, 0],
                        data[frame, feat_index, 1],
                        data[frame, feat_index, 2] + 0.1,  # slight offset for readability
                        coord_text,
                        color="red",
                        fontsize=10,
                    )

    ani = FuncAnimation(fig, update, frames=len(data), interval=1000 / fps, repeat=False)
    ani.save(path, writer="ffmpeg", fps=fps)
    plt.close()


def output_path_callback(ctx: typer.Context, value: Path):
    if "idx" in value.stem:
        return value.with_stem(value.stem.replace("idx", str(ctx.params["idx"])))
    else:
        return value


def input_path_callback(ctx: typer.Context, value: Path):
    if value is None:
        return value
    else:
        data_path = value / str(ctx.params["idx"])
        if data_path.exists():
            return data_path
        else:
            raise typer.BadParameter(f"Input data file {data_path} does not exist.")


def plot_example(
    idx: int = 0,
    token: Annotated[
        Optional[str],
        typer.Argument(envvar="HF_TOKEN", help="The HF token to download the dataset."),
    ] = None,
    output_file: Path = typer.Option(
        Path("sample_idx_vis_val_dasyd_time-qa.mp4"), callback=output_path_callback
    ),
    paint_features: Annotated[
        Optional[list[str]],
        typer.Option(
            help=f"List of features to paint in the visualization. Possible values are\n{HML_JOINT_NAMES}"
        ),
    ] = None,
    normalize: bool = typer.Option(True, help="Normalize the motion data"),
    labels: bool = typer.Option(False, help="Display labels for the painted features"),
    label_vector: bool = typer.Option(False, help="Display labels for the painted features"),
    display_vector: bool = typer.Option(
        False, help="Display vector from the root joint to the painted features"
    ),
    ask_for_token: bool = typer.Option(
        True, help="Ask for the HF token if not found in the environment variable"
    ),
    input_path: Optional[Path] = typer.Option(
        None,
        callback=input_path_callback,
        help="Instead of using the huggingface dataset, provide a custom data file. This should point to the folder with all the data directories per ID in it.",
    ),
):
    if paint_features is None:
        paint_features = []
    else:
        for feature in paint_features:
            if feature not in HML_JOINT_NAMES:
                raise typer.BadParameter(
                    f"Feature {feature} is not a valid joint name. Possible values are\n{HML_JOINT_NAMES}"
                )

    # Get the data
    if input_path is not None:
        with open(input_path / "data.json") as f:
            meta_data = json.load(f)
        meta_data["input_from"] = input_path

        with open_file(input_path / "data.hdf5", "r") as hdf5_file:
            sample = hdf5_file.root.motion.motion[:][0, ...].astype("float32")

    else:
        if token is None and ask_for_token:
            token = typer.prompt("Insert your HF token", hide_input=True)

        huggingface_repo = "dasyd/time-qa"
        huggingfae_split = "binary"

        def _load_dataset_split(splits: list[str]):
            """Workaround to overcome the missing hf implementation of only dowloading the split shards"""
            return load_dataset(
                huggingface_repo,
                huggingfae_split,
                data_dir=huggingfae_split,
                data_files={f"{split}-*" for split in splits},
                verification_mode=VerificationMode.NO_CHECKS,
                num_proc=len(splits),
                token=token,
            )

        ds = _load_dataset_split(["val"]).with_format("numpy")
        meta_data["input_from"] = f"{huggingface_repo} split {huggingfae_split}"

        full_sample = ds["val"][idx]

        sample = full_sample["trajectory"]
        meta_data = full_sample.copy()
        del meta_data["trajectory"]

    typer.echo("")

    table = Table("Key", "Value")
    for key, value in meta_data.items():
        table.add_row(key, str(value))
    table.add_row("animation_file", str(output_file))

    Console().print(table)

    plot_3d_motion(
        output_file,
        t2m_kinematic_chain,
        sample.transpose(2, 0, 1),  # trajectory 300 22 3
        normalize=normalize,
        painting_features=paint_features,
        labels=labels,
        label_vector=label_vector,
        display_vector=display_vector,
    )
