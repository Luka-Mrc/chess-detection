# Chess Detection

Automatic chess position detection from images. Detects the board and pieces, outputs a FEN string.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev]"
```

## Usage

**Full pipeline Flask App** (board detection + pieces + FEN):
```bash
python app.py
```
Starts a local server at `http://localhost:5000`.

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Web UI |
| `/detect` | POST | Detect position, return FEN |

`POST /detect` form parameters:

| Parameter | Type | Values | Default |
|-----------|------|---------|---------|
| `image` | file | any image | required |
| `method` | string | `canny`, `hough`, `dnn` | `canny` |

Response:
```json
{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "success": true}
```

**Full pipeline cli** (board detection + pieces + FEN):
```bash
python scripts/detect.py --image path/to/image.jpg --output path/to/output.jpg
```

**Piece detection only** (no board, no FEN):
```bash
python scripts/detect.py --image path/to/image.jpg --no-board
```

**Custom configs:**
```bash
python scripts/detect.py --image path/to/image.jpg \
  --board-config config/board_detection.yaml \
  --piece-config config/piece_detection.yaml \
  --output path/to/output.jpg
```

## Structure

- `src/chess_detection/` - library (board detection, piece detection, pipeline)
- `scripts/` - training, evaluation, and inference entry points
- `config/` - hyperparameters (YAML)
- `data/` - images, annotations, patches (not committed to git)
