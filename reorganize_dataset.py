"""
Dataset reorganization script:
1. Fix label mapping: normal=non-fire, abnormal=fire
2. Create class-based subfolders in both
3. Rename all files to English (no Korean characters)
"""

import os
import shutil
from pathlib import Path

BASE = Path(r'C:\fireimage_detection\data\fireimage')

CLASS_MAP = {
    '구름': 'cloud',
    '안개연무': 'fog_haze',
    '굴뚝연기': 'chimney_smoke',
    '백색회색연기': 'white_gray_smoke',
    '화염': 'flame',
    '흑색연기': 'black_smoke',
}

LABELLING_FOLDER_MAP = {
    'NegativeDB_구름': ('normal', 'cloud'),
    'NegativeDB_굴뚝연기': ('normal', 'chimney_smoke'),
    'NegativeDB_안개연무': ('normal', 'fog_haze'),
    'PositiveRealDB_백색회색연기': ('abnormal', 'white_gray_smoke'),
    'PositiveRealDB_화염': ('abnormal', 'flame'),
    'PositiveRealDB_흑색연기': ('abnormal', 'black_smoke'),
    'PositiveSyntheticDB_백색회색연기': ('abnormal', 'white_gray_smoke'),
    'PositiveSyntheticDB_화염': ('abnormal', 'flame'),
    'PositiveSyntheticDB_흑색연기': ('abnormal', 'black_smoke'),
}

LABELLING_TYPE_MAP = {
    'NegativeDB_구름': None,
    'NegativeDB_굴뚝연기': None,
    'NegativeDB_안개연무': None,
    'PositiveRealDB_백색회색연기': 'real',
    'PositiveRealDB_화염': 'real',
    'PositiveRealDB_흑색연기': 'real',
    'PositiveSyntheticDB_백색회색연기': 'syn',
    'PositiveSyntheticDB_화염': 'syn',
    'PositiveSyntheticDB_흑색연기': 'syn',
}

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def parse_flat_file(name):
    stem = Path(name).stem
    parts = stem.split('_')
    db_type = parts[0]
    cls_kr = parts[1] if len(parts) > 1 else 'unknown'
    cls_en = CLASS_MAP.get(cls_kr, cls_kr)
    if db_type == 'PositiveRealDB':
        type_tag = 'real'
        dest_folder = 'abnormal'
    elif db_type == 'PositiveSyntheticDB':
        type_tag = 'syn'
        dest_folder = 'abnormal'
    else:  # NegativeDB
        type_tag = None
        dest_folder = 'normal'
    return dest_folder, cls_en, type_tag

def collect_flat_files(folder):
    src_dir = BASE / folder
    result = []
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            result.append(f)
    return result

def build_new_name(cls_en, type_tag, idx):
    if type_tag:
        return f'{cls_en}_{type_tag}_{idx:06d}.jpg'
    return f'{cls_en}_{idx:06d}.jpg'

# ─── STEP 1: Reorganize normal/ and abnormal/ ─────────────────────────────────
print('=' * 60)
print('STEP 1: Reorganizing normal/ and abnormal/')
print('=' * 60)

# Collect all flat files from both folders
all_flat = []
for folder in ('normal', 'abnormal'):
    for f in collect_flat_files(folder):
        dest_folder, cls_en, type_tag = parse_flat_file(f.name)
        all_flat.append((f, dest_folder, cls_en, type_tag))

# Write to a temp staging area first (avoids read/write conflict on the same dir)
TEMP = BASE / '_temp_reorg'
if TEMP.exists():
    shutil.rmtree(TEMP)

counters = {}
for src, dest_folder, cls_en, type_tag in all_flat:
    key = (dest_folder, cls_en, type_tag)
    counters[key] = counters.get(key, 0) + 1
    new_name = build_new_name(cls_en, type_tag, counters[key])
    dest_dir = TEMP / dest_folder / cls_en
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / new_name)

# Remove old flat files from both folders
for folder in ('normal', 'abnormal'):
    for f in (BASE / folder).iterdir():
        if f.is_file():
            f.unlink()

# Move from temp to final
for folder in ('normal', 'abnormal'):
    temp_dir = TEMP / folder
    if not temp_dir.exists():
        continue
    for cls_dir in temp_dir.iterdir():
        final_dir = BASE / folder / cls_dir.name
        final_dir.mkdir(parents=True, exist_ok=True)
        for f in cls_dir.iterdir():
            shutil.move(str(f), str(final_dir / f.name))

shutil.rmtree(TEMP)

print('normal/ and abnormal/ reorganized.')
for folder in ('normal', 'abnormal'):
    for cls_dir in sorted((BASE / folder).iterdir()):
        if cls_dir.is_dir():
            cnt = len([x for x in cls_dir.iterdir() if x.is_file()])
            print(f'  {folder}/{cls_dir.name}/: {cnt} files')

# ─── STEP 2: Reorganize labelling/ ────────────────────────────────────────────
print()
print('=' * 60)
print('STEP 2: Reorganizing labelling/')
print('=' * 60)

LABEL_SRC = BASE / 'labelling'
LABEL_TEMP = BASE / '_temp_labelling'
if LABEL_TEMP.exists():
    shutil.rmtree(LABEL_TEMP)

counters_l = {}
for old_folder_name, (dest_folder, cls_en) in LABELLING_FOLDER_MAP.items():
    type_tag = LABELLING_TYPE_MAP[old_folder_name]
    old_dir = LABEL_SRC / old_folder_name
    if not old_dir.exists():
        continue
    # Files may be in location-based subdirectories
    for img_file in old_dir.rglob('*'):
        if img_file.is_file() and img_file.suffix.lower() in IMG_EXTS:
            key = (dest_folder, cls_en, type_tag)
            counters_l[key] = counters_l.get(key, 0) + 1
            new_name = build_new_name(cls_en, type_tag, counters_l[key])
            dest_dir = LABEL_TEMP / dest_folder / cls_en
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, dest_dir / new_name)

# Remove old labelling contents
for item in LABEL_SRC.iterdir():
    if item.is_dir():
        shutil.rmtree(item)
    elif item.is_file():
        item.unlink()

# Move from temp to labelling/
for folder in ('normal', 'abnormal'):
    temp_dir = LABEL_TEMP / folder
    if not temp_dir.exists():
        continue
    for cls_dir in temp_dir.iterdir():
        final_dir = LABEL_SRC / folder / cls_dir.name
        final_dir.mkdir(parents=True, exist_ok=True)
        for f in cls_dir.iterdir():
            shutil.move(str(f), str(final_dir / f.name))

shutil.rmtree(LABEL_TEMP)

print('labelling/ reorganized.')
for folder in ('normal', 'abnormal'):
    for cls_dir in sorted((LABEL_SRC / folder).iterdir()):
        if cls_dir.is_dir():
            cnt = len([x for x in cls_dir.iterdir() if x.is_file()])
            print(f'  labelling/{folder}/{cls_dir.name}/: {cnt} files')

# ─── STEP 3: Final summary ────────────────────────────────────────────────────
print()
print('=' * 60)
print('FINAL STRUCTURE')
print('=' * 60)
total = 0
for top in ('normal', 'abnormal'):
    for cls_dir in sorted((BASE / top).iterdir()):
        if cls_dir.is_dir():
            cnt = len([x for x in cls_dir.iterdir() if x.is_file()])
            total += cnt
            label = 'Non-fire' if top == 'normal' else 'Fire'
            print(f'  [{label}] {top}/{cls_dir.name}/: {cnt} files')
print(f'\nTotal training images: {total}')
