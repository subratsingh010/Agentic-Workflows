import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


def main() -> None:
    output_path = Path("openapi.json")
    output_path.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

