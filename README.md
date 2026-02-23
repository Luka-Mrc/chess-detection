# Chess Detection

Automatic chess position detection from images. Detects the board and pieces, outputs a FEN string.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev]"
```

## Piece Detection Usage

```bash
python scripts/detect.py --image path/to/image.jpg --output path/to/output/image.jpg
```

## Structure

- `src/chess_detection/` - library (board detection, piece detection, pipeline)
- `scripts/` - training, evaluation, and inference entry points
- `config/` - hyperparameters (YAML)
- `data/` - images, annotations, patches (not committed to git)
