from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from subprocess import DEVNULL, check_call

import typer
from rich.console import Console
from rich.progress import track


def check_dataset_path(ctx: typer.Context, dataset_path: Path):
    if not dataset_path.exists():
        raise typer.BadParameter(f"Dataset path {dataset_path} does not exist.")

    return dataset_path.absolute()


def transcode_animations(
    dataset_path: Path = typer.Option(
        help=(
            "Path to the text prompts dataset. "
            "Expected to contain folders of any names and the input video files within them. "
            "These folders are where the results will be stored."
        ),
        callback=check_dataset_path,
    ),
    n_parallel: int = 10,
    file_in: str = "render_smpl.mp4",
    file_out: str = "render_smpl_compressed.mp4",
):
    # Share access to the console between all threads
    console = Console()

    all_instance_files = list(dataset_path.glob(f"*/{file_in}"))
    all_instance_files.sort()

    # Remove all files where the output already exists
    missing_instances = [f for f in all_instance_files if not (f.parent / file_out).exists()]

    console.print(
        f"Missing {len(all_instance_files)} instances ({len(missing_instances) * 100 / len(all_instance_files):.2f}%)"
    )

    if not missing_instances:
        console.print("Nothing to render")
        return

    def process_single(video_in: Path) -> None:
        check_call(
            [
                "ffmpeg",
                "-i",
                file_in,
                "-c:v",
                "vp9",
                "-strict",  # Follow standars more strictly
                "strict",
                file_out,
            ],
            stdout=DEVNULL,
            stderr=DEVNULL,
            cwd=video_in.parent,
        )

    with ThreadPool(processes=n_parallel) as pool:
        for _ in track(
            pool.imap_unordered(process_single, missing_instances),
            total=len(missing_instances),
            console=console,
        ):
            pass
