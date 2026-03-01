"""
Creates mini_bdd100k.zip — 100 train + 20 val samples per class.
Reads directly from the original archive.zip without full extraction.

Usage:
    python create_mini_dataset.py
"""

import json, zipfile, io, os
from collections import defaultdict

SRC_ZIP   = '/home/vamsikrishna/Computer_Vison_RoadMap/applied_cv/archive.zip'
OUT_ZIP   = '/home/vamsikrishna/Computer_Vison_RoadMap/applied_cv/mini_bdd100k.zip'
CLASSES   = ['car','traffic sign','traffic light','person',
             'truck','bus','bike','rider','motor','train']
TRAIN_N   = 100   # images per class for training
VAL_N     = 20    # images per class for validation

print(f'Opening {SRC_ZIP} ...')
src = zipfile.ZipFile(SRC_ZIP, 'r')
all_names = set(src.namelist())

# ── helpers ────────────────────────────────────────────────────────────────────
def read_json_from_zip(zf, path):
    with zf.open(path) as f:
        return json.load(f)

def get_classes(entry):
    return {lbl['category']
            for lbl in entry.get('labels', [])
            if lbl.get('box2d') and lbl['category'] in CLASSES}

def select_samples(entries, image_prefix, n_per_class):
    """Return (selected_entries, filtered_labels_list)."""
    bucket   = defaultdict(list)   # class → [entry]
    selected = {}                  # name → entry

    for entry in entries:
        img_path = image_prefix + entry['name']
        if img_path not in all_names:
            continue
        cats = get_classes(entry)
        for cat in cats:
            if len(bucket[cat]) < n_per_class:
                bucket[cat].append(entry)
                selected[entry['name']] = entry

    print(f'  Per-class counts:')
    for cls in CLASSES:
        print(f'    {cls:<15} {len(bucket[cls])}')
    print(f'  Unique images selected: {len(selected)}')
    return list(selected.values())

# ── read label JSONs from zip ──────────────────────────────────────────────────
TRAIN_JSON_PATH = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
VAL_JSON_PATH   = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'

print('\nReading train labels ...')
train_entries = read_json_from_zip(src, TRAIN_JSON_PATH)
print(f'  {len(train_entries):,} entries in train JSON')

print('\nReading val labels ...')
val_entries = read_json_from_zip(src, VAL_JSON_PATH)
print(f'  {len(val_entries):,} entries in val JSON')

# ── select samples ─────────────────────────────────────────────────────────────
TRAIN_PREFIX = 'bdd100k/bdd100k/images/100k/train/'
VAL_PREFIX   = 'bdd100k/bdd100k/images/100k/val/'

print(f'\nSelecting {TRAIN_N} train samples per class ...')
mini_train = select_samples(train_entries, TRAIN_PREFIX, TRAIN_N)

print(f'\nSelecting {VAL_N} val samples per class ...')
mini_val = select_samples(val_entries, VAL_PREFIX, VAL_N)

# ── write mini zip ─────────────────────────────────────────────────────────────
print(f'\nWriting {OUT_ZIP} ...')
total = len(mini_train) + len(mini_val)

with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as dst:

    # Train images
    for i, entry in enumerate(mini_train, 1):
        zip_path = TRAIN_PREFIX + entry['name']
        data = src.read(zip_path)
        dst.writestr(zip_path, data)
        if i % 100 == 0 or i == len(mini_train):
            print(f'  Train images: {i}/{len(mini_train)}', end='\r')
    print()

    # Val images
    for i, entry in enumerate(mini_val, 1):
        zip_path = VAL_PREFIX + entry['name']
        data = src.read(zip_path)
        dst.writestr(zip_path, data)
        if i % 50 == 0 or i == len(mini_val):
            print(f'  Val images  : {i}/{len(mini_val)}', end='\r')
    print()

    # Filtered train labels JSON
    dst.writestr(TRAIN_JSON_PATH, json.dumps(mini_train))

    # Filtered val labels JSON
    dst.writestr(VAL_JSON_PATH, json.dumps(mini_val))

src.close()

size_mb = os.path.getsize(OUT_ZIP) / 1e6
print(f'\n✅ Done!')
print(f'   Train images : {len(mini_train)}')
print(f'   Val images   : {len(mini_val)}')
print(f'   Output size  : {size_mb:.0f} MB')
print(f'   Saved to     : {OUT_ZIP}')
