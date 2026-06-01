# Data notes (local setup)

This repository expects the dataset to be available under `data/` with the following convention:

## Expected paths
- Images: `data/images/`
- Raw/processed JSON files: see subfolders such as:
  - `data/raw-data/`
  - `data/final-data/`

## Image download (Google Drive)

Images are **not** stored in Git. Download the data folder (include final-data/ and images/)from the shared Drive and place it at:
```
data/images/
```

**Drive link:** <https://drive.google.com/drive/folders/1ed5XsY8IuK1pPSdsLp9Fd6TMsy4-tVhY?usp=sharing>

## Verification
After downloading, you should see files like:
```
data/images/post00001.jpg
data/images/post01657.webp
...
```

If you change image formats (e.g., convert to `.pdf` for the report), keep a copy in `report/figures/` and do not overwrite the raw dataset images unless you also update all paths consistently.

