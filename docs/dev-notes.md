# Website Development Notes

## Local preview

The static project page is self-contained (no build step, all CSS/JS vendored under `docs/static/`). Preview it locally with:

```shell
python -m http.server -d docs 8080
```

## Publishing

GitHub Pages publishes the `/docs` directory. The entry point is `docs/index.html`, with its assets under `docs/static/`.

## Updating results

The result tables in `index.html` report held-out test scores. Display accuracy, precision, recall,
and F1 as percentages; keep ROUGE, METEOR, and normalized LLMJudge on the 0–1 scale.
For Q2, Q3, and Naive, precision, recall, and F1 are macro-averaged, and ROUGE is stemmed ROUGE-L F1.
Keep the Human, ChatTS, and xQA reference scores under their respective evaluation setups;
do not mark cross-setup winners without reconciling the evaluation conditions.

The retention table compares each adapter with the base at the same precision.
PPL/base is a multiplicative ratio, not a percentage or a QuAnTS task metric.

Check every displayed value against the aggregate result CSVs before publishing.
Keep generated predictions and model artifacts outside the website repository.

## Adding new samples

To show a different sample, render its video from the dataset folder, burning in the clock the questions refer to:

```shell
ID=27500
ffmpeg -i generated-dataset-30_000/data/$ID/render_smpl_compressed.mp4 \
    -vf "fps=10,scale=320:320,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
expansion=normal:text='%{pts \: hms}':fontcolor=white:fontsize=18:x=(w-text_w)/2:y=h-th-10:\
box=1:boxcolor=black:boxborderw=5,pad=320:364:0:0:color=white" \
    -movflags +faststart -pix_fmt yuv420p -crf 30 -an docs/static/videos/$ID.mp4
ffmpeg -ss 2 -i docs/static/videos/$ID.mp4 -frames:v 1 -q:v 6 docs/static/videos/$ID.jpg
```

The bottom padding keeps the timestamp clear of the video controls, and the JPEG serves as the poster frame. Then update the corresponding `.sample-chip` and `.sample-panel` blocks in `index.html`.
Only show samples whose answers you have verified against the action sequence.
