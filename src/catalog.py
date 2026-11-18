"""Build a read-only SQLite catalog from the packaged validation inventory."""

from contextlib import closing
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile

from .evaluate import ROOT

SOURCE = ROOT / "data/validation/catalog.json"
FILTER_FIELDS = ("garment_type", "color", "style")
SCHEMA = """
CREATE TABLE category (
    code TEXT PRIMARY KEY NOT NULL
);
CREATE TABLE garment (
    sku TEXT PRIMARY KEY NOT NULL,
    category_code TEXT NOT NULL REFERENCES category(code),
    garment_type TEXT NOT NULL,
    color TEXT NOT NULL,
    style TEXT NOT NULL,
    image_path TEXT NOT NULL UNIQUE
);
CREATE INDEX garment_filters ON garment(category_code, garment_type, color, style);
"""


@lru_cache(maxsize=1)
def catalog_db(source=SOURCE, output_dir=ROOT / "runs"):
    payload = source.read_bytes()
    digest = hashlib.sha256(SCHEMA.encode() + payload).hexdigest()[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"catalog-{digest}.sqlite3"
    if path.exists():
        return path
    items = json.loads(payload)
    with NamedTemporaryFile(dir=output_dir, suffix=".sqlite3", delete=False) as file:
        temp = Path(file.name)
    try:
        with closing(sqlite3.connect(temp)) as db, db:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(SCHEMA)
            db.executemany("INSERT INTO category VALUES (?)", [(c,) for c in sorted({r["category"] for r in items})])
            db.executemany("INSERT INTO garment VALUES (?, ?, ?, ?, ?, ?)", [
                (r["sku"], r["category"], r["garment_type"], r["color"], r["style"], r["image_path"])
                for r in items
            ])
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def candidate_skus(filters):
    fields = ("category", *FILTER_FIELDS)
    columns = ("category_code", *FILTER_FIELDS)
    active = [(col, filters[field]) for field, col in zip(fields, columns) if field in filters]
    conditions = " AND ".join(f"{column} = ?" for column, _ in active)
    with closing(sqlite3.connect(catalog_db().as_uri() + "?mode=ro", uri=True)) as db:
        return [sku for (sku,) in db.execute(
            f"SELECT sku FROM garment WHERE {conditions} ORDER BY sku",
            [value for _, value in active],
        )]
