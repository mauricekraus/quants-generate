from typer import Typer

from .prompts import generate as generate_prompts
from .render import plot_example

app = Typer()
app.command("render")(plot_example)
app.command("prompts")(generate_prompts)

if __name__ == "__main__":
    app()
