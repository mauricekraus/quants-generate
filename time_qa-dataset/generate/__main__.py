from typer import Typer

from .animate_stmc import animate_motion
from .generate_motion_stmc import generate_motion
from .prompts import generate as generate_prompts
from .render_priormdm import plot_example

app = Typer()
app.command("prompts")(generate_prompts)
app.command("render-priormdm")(plot_example)
app.command("generate-motion-stmc")(generate_motion)
app.command("render-stmc")(animate_motion)


if __name__ == "__main__":
    app()
