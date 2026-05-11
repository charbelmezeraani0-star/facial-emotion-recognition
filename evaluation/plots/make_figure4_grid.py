"""
Run this script AFTER placing your 4 deployment screenshots in this same folder:
  - deploy_happy_box.png   — webcam/detection frame showing happy face with yellow bounding box
  - deploy_happy_bar.png   — Streamlit probability bar showing ~100% happy
  - deploy_angry_box.png   — webcam/detection frame showing angry face with red bounding box
  - deploy_angry_bar.png   — Streamlit probability bar showing ~99.9% angry

Then run:
  python3 evaluation/plots/make_figure4_grid.py

Output: evaluation/plots/figure4_deployment.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

PLOTS = Path(__file__).parent

files = {
    'top_left':     PLOTS / 'deploy_happy_box.png',
    'top_right':    PLOTS / 'deploy_happy_bar.png',
    'bottom_left':  PLOTS / 'deploy_angry_box.png',
    'bottom_right': PLOTS / 'deploy_angry_bar.png',
}

missing = [k for k, v in files.items() if not v.exists()]
if missing:
    print('Missing files:', missing)
    print('Place the screenshots in evaluation/plots/ with the names above, then re-run.')
    raise SystemExit(1)

fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
fig.patch.set_facecolor('white')

labels = [
    ('(a) Happy detection — bounding box output',  'top_left'),
    ('(b) Happy prediction — probability bars',    'top_right'),
    ('(c) Angry detection — bounding box output',  'bottom_left'),
    ('(d) Angry prediction — probability bars',    'bottom_right'),
]

positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for (lbl, key), (r, c) in zip(labels, positions):
    ax = axes[r][c]
    img = mpimg.imread(str(files[key]))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(lbl, fontsize=9.5, color='#333333', pad=4)

plt.suptitle('Figure 4: Deployed Streamlit Application — Happy and Angry Emotion Detection',
             fontsize=11, fontweight='bold', color='#1F4E79', y=1.01)
plt.tight_layout(pad=0.8)
plt.savefig(str(PLOTS / 'figure4_deployment.png'),
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: evaluation/plots/figure4_deployment.png')
