import typer

from miles.utils.ft.cli.launch import launch

app = typer.Typer(help="Miles Fault Tolerance CLI.")
app.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)(launch)
