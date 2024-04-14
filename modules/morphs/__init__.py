from pathlib import Path
MODEL_MORPHS = [fp.stem for fp in Path(__file__).parent.iterdir() if fp.is_dir() and not fp.name.startswith('_')]
