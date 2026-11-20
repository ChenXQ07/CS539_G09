"""Write fixed data specifications without silently changing existing versions."""
from pathlib import Path


def write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to replace a frozen artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as target:
        target.write(content)
