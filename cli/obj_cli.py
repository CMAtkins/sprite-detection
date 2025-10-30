import os
import json
import sqlite3
import requests
import typer
from rich.console import Console
from rich.table import Table
from PIL import Image
from io import BytesIO
from datetime import datetime
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import fnmatch

app = typer.Typer(help="OBJ: Command-line toolkit for object detection and analysis")
config_app = typer.Typer(help="Manage OBJ configuration (API URL, DB path, etc)")
app.add_typer(config_app, name="config")

console = Console()

CONFIG_DIR = Path.home() / ".obj"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "api_url": "http://127.0.0.1:8000",
    "db_path": str(Path.cwd() / "obj_local.db")     # TODO
}


# Load config file or create default
def load_config() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(cfg: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)


config = load_config()
API_URL = config["api_url"]
DB_PATH = os.path.expanduser(config["db_path"])


# Show current config
@config_app.command("show")
def show_config():
    console.print(json.dumps(load_config(), indent=4))


# Set a config value
@config_app.command("set")
def set_config(key: str, value: str):
    cfg = load_config()
    normalized_key = key.replace("-", "_")
    if normalized_key not in cfg:
        console.print(f"[red]Unknown config key:[/red] {key}")
        console.print(f"Available keys: {', '.join(cfg.keys())}")
        raise typer.Exit(1)
    cfg[normalized_key] = value
    save_config(cfg)
    console.print(f"[green]Updated {key} →[/green] {value}")


# Reset config
@config_app.command("reset")
def reset_config():
    save_config(DEFAULT_CONFIG)
    console.print("[yellow]Configuration reset to defaults.[/yellow]")


# Upload an image or folder of images to the FastAPI server for object detection
@app.command()
def detect(
    path: str,
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively search for images in subdirectories."
    ),
    pattern: str = typer.Option(
        "*.jpg",
        "--pattern",
        "-p",
        help="File name pattern to match (e.g., '*.jpg', '*.png', '*_FGCU_*.jpg')."
    ),
):

    # TODO: Init (DB)

    cfg = load_config()
    api_url = cfg["api_url"]

    # Collect image paths
    image_paths = []
    if os.path.isfile(path):
        image_paths = [path]
    elif os.path.isdir(path):
        walker = os.walk(path) if recursive else [(path, [], os.listdir(path))]
        for root, _, files in walker:
            for file in files:
                if fnmatch.fnmatch(file.lower(), pattern.lower()):
                    image_paths.append(os.path.join(root, file))
    else:
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise typer.Exit(1)

    if not image_paths:
        console.print(f"[yellow]No matching images found for pattern:[/yellow] '{pattern}' in {path}")
        raise typer.Exit(0)

    console.print(f"[cyan]Preparing batch upload for {len(image_paths)} image(s)...[/cyan]\n")

    # Multipart POST
    upload_list = [("files", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in image_paths]

    # Upload all images in one request with progress bar
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Uploading batch...", total=1)
        try:
            response = requests.post(f"{api_url}/batch_detect", files=upload_list)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Request failed:[/red] {e}")
            raise typer.Exit(1)
        finally:
            # Avoid descriptor leak
            for _, ftuple in upload_list:
                ftuple[1].close()
        progress.advance(task)

    # Parse server response
    data = response.json()
    count = data.get("count", 0)
    detections = data.get("detections", [])
    zip_url = data.get("zip_url")
    grid_urls = data.get("grid_urls", [])

    console.print(f"\n[green]✅ Completed batch detection for {count} image(s)[/green]")

    # for detection in detections:
    # TODO: Store detections (DB)

    # Show results summary
    if zip_url:
        console.print(f"[blue]📦 Results ZIP:[/blue] {api_url.rstrip('/')}{zip_url}")
    if grid_urls:
        console.print(f"[blue]🖼️ Grid images:[/blue]")
        for g in grid_urls:
            console.print(f"   {api_url.rstrip('/')}{g}")
    console.print()


@app.command()
def stats():
    # TODO: Show a quick summary of stored detections (DB)
    print("Under construction 🚀")


@app.command()
def show(image_name: str):
    # TODO: Display annotated image from a previously stored detection (DB)
    print("Under construction 🚀")


@app.command()
def query(question: str):
    # TODO: Ask natural-language question about stored detections
    print("Under construction 🚀")


if __name__ == "__main__":
    app()
