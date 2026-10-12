import sys

from typer import Typer

from .prompts import generate as generate_prompts
from .transcode_animations import transcode_animations

app = Typer()
app.command("prompts")(generate_prompts)
app.command("transcode-animations")(transcode_animations)

try:
    from .animate_stmc import animate_motion
    from .generate_motion_stmc import generate_motion
except ImportError as error:
    print(
        f"Warning: The STMC dependencies are missing ({error}), so the commands 'generate-motion-stmc' and "
        "'render-stmc' are unavailable. See the installation instructions in the README: "
        "https://github.com/mauricekraus/quants-generate#installation",
        file=sys.stderr,
    )
else:
    app.command("generate-motion-stmc")(generate_motion)
    app.command("render-stmc")(animate_motion)


if __name__ == "__main__":
    app()
