# Molinaro → LIMU-BERT Fine-tune Pipeline

End-to-end recipe for fine-tuning the pretrained LIMU-BERT model on your full Molinaro
dataset, with sensor-frame alignment to your "self" data so that downstream evaluation
on self transfers cleanly.

## Pipeline overview

```
[Your full Molinaro CSVs]                          [dataset_self.csv]
        │                                                   │
        │  ┌──────────────────────────────────┐             │
        │  │ compute_alignment.py             │ ◄───────────┘
        │  │   - estimate gravity in each     │
        │  │   - build 3x3 rotation R         │
        │  └──────────────────────────────────┘
        │              │
        ▼              ▼
┌──────────────────────────────────────┐
│ preprocess_molinaro.py               │
│   - extract LEFT thigh (6 features)  │
│   - apply rotation R                 │
│   - 200 Hz → 20 Hz (anti-aliased)    │
│   - 120-sample windows, no overlap   │
│   - majority-vote labels per window  │
│   - drop low-purity windows          │
└──────────────────────────────────────┘
        │
        ▼
LIMU-BERT-Public/dataset/molinaro/
  data_20_120.npy       (N, 120, 6)
  label_20_120.npy      (N, 120, 1)
  label_map.json
        │
        ▼  (then run official LIMU-BERT scripts)
┌─────────────────────────────────────────────────┐
│ embedding.py  — load pretrained LIMU-BERT,      │
│                 generate (N, 120, H) embeddings │
│ classifier.py — freeze BERT, train GRU on labels│
└─────────────────────────────────────────────────┘
```

## Step 0 — Setup

```bash
git clone https://github.com/dapowan/LIMU-BERT-Public
cd LIMU-BERT-Public
pip install -r requirements.txt
pip install scipy pandas      # for the preprocessing scripts

# Drop these three files into the repo root (or anywhere on PYTHONPATH):
#   preprocess_molinaro.py
#   compute_alignment.py
#   README.md
```

## Step 1 — Compute the gravity-alignment rotation

Pick **one** Molinaro CSV that contains some still moments (standing / sitting). Then:

```bash
python compute_alignment.py \
    --self_csv     /path/to/dataset_self.csv \
    --molinaro_csv /path/to/molinaro/some_subject.csv \
    --out          gravity_R.npy
```

The script estimates each frame's gravity vector from the 20% lowest-gyro samples,
then builds the 3×3 matrix `R` such that `R @ g_molinaro ≈ g_self`. Inspect the
console output — direction error after rotation should be < 5 deg. If it's larger,
your reference files likely don't contain enough still time; try a different file.

## Step 2 — Preprocess all Molinaro CSVs into LIMU-BERT format

```bash
python preprocess_molinaro.py \
    --input_dir   /path/to/molinaro_csvs/ \
    --output_dir  ./dataset/molinaro/ \
    --label_col   activity \
    --src_fs      200 \
    --tgt_fs      20 \
    --window_size 120 \
    --purity      0.8 \
    --rotation    gravity_R.npy
```

Replace `--label_col activity` with whatever your label column is actually called.
The script will print the discovered label vocabulary and ask you to add an entry to
`dataset/data_config.json`. Copy-paste the printed JSON block into that file.

Output:

- `dataset/molinaro/data_20_120.npy`     — shape `(N, 120, 6)` float32
- `dataset/molinaro/label_20_120.npy`    — shape `(N, 120, 1)` int32
- `dataset/molinaro/label_map.json`      — class-name → int-id mapping

## Step 3 — Add the dataset to LIMU-BERT's config

Open `dataset/data_config.json` and paste the entry that `preprocess_molinaro.py`
printed. It will look something like:

```json
"molinaro_20_120": {
    "sr": 20,
    "seq_len": 120,
    "dimension": 6,
    "activity_label_index": 0,
    "activity_label_size":  7,
    "activity_label":       ["walking", "stair_up", "stair_down", "sitting", "standing", "running", "jumping"],
    "size":                 1234
}
```

Also register the dataset in `config.py` (look at how `hhar`, `motion`, etc. are
listed and add `molinaro` next to them).

## Step 4 — Generate embeddings using the pretrained LIMU-BERT

The repo ships pretrained checkpoints under `saved/`. Pick the one trained on the
**most similar** dataset (or use the merged checkpoint if available). For a generic
HAR transfer, the merged or `motion` checkpoint is a reasonable starting point.

```bash
python embedding.py v1 molinaro 20_120 -f <pretrained_checkpoint_name>
```

This saves `embed/embed_<name>_molinaro_20_120.npy` — the (N, 120, H) features.

## Step 5 — Fine-tune the classifier (LIMU-BERT frozen)

```bash
python classifier.py v2 molinaro 20_120 \
    -f <pretrained_checkpoint_name> \
    -s molinaro_gru_v1 \
    -l 0          # label index 0 = activity
```

This trains the GRU classifier on top of the frozen embeddings. Adjust the
`label_rate` and `balance` parameters in the `main` of `classifier.py` to control
how much labeled data is used and whether classes are balanced.

## Notes & gotchas

- **Gravity alignment recovers only 2 DOF.** The residual rotation around the
  gravity axis cannot be determined from static data. For HAR this rarely matters,
  but if your downstream task is direction-sensitive (e.g. left-vs-right turn
  detection), consider also adding small random-rotation augmentation during
  classifier training.
- **Acc/gyro share one IMU body frame**, so they get rotated by the SAME R. The
  preprocessing script does this automatically.
- **Window purity filter** (`--purity 0.8`) drops transition windows where the
  activity label changes mid-window. Lower this to 0.6 if you're losing too much
  data; raise it to 1.0 to keep only fully-clean windows.
- **Anti-aliased downsample** uses `scipy.signal.resample_poly` (polyphase) —
  proper FIR filter applied before decimation so the 20 Hz output isn't aliased.
- **The pretrained LIMU-BERT was trained on phone IMUs at waist/pocket placements.**
  Your thigh placement is similar enough that transfer should work, but you may see
  larger gains from continuing pretraining on Molinaro before training the
  classifier — see the optional "continued pretrain" path below.

## Optional: continued self-supervised pretraining on Molinaro

If you want LIMU-BERT to better adapt to the thigh-mount domain before training the
classifier, run masked-reconstruction pretraining on your unlabeled Molinaro data
(or all of it — the masking task doesn't use labels):

```bash
python pretrain.py v1 molinaro 20_120 -s limu_molinaro_v1 \
    -f <starting_checkpoint>      # warm-start from existing pretrained model
```

Then use `limu_molinaro_v1` as the checkpoint in steps 4 and 5.

## Once your "self" data is labeled

When you finish manually labeling your self data, run the same preprocessing on it
(adjust column names — use the included `preprocess_self.py` template, or pass
custom `--feature_cols` if you parameterize the script). **Do NOT** apply the R
matrix to self data — R rotates Molinaro INTO self's frame, so self stays as-is.
Then evaluate the trained `molinaro_gru_v1` classifier on self's preprocessed npy.
