import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define data
data = {
    'FPS': [
        28.5, 24, 21.7, 20, 14, 7.2,  # yolo11s
        24, 18.5, 9.5,               # yolo11n
        28, 25, 17                   # Edge
    ],
    'MOTA': [
        16.4, 16.1, 16.4, 18.1, 19.1, 22, # yolo11s
        17.4, 17.7, 21.2,                 # yolo11n
        21, 23.5, 30.1                    # Edge
    ],
    'Group': [
        'yolo11s', 'yolo11s', 'yolo11s', 'yolo11s', 'yolo11s', 'yolo11s',
        'yolo11n', 'yolo11n', 'yolo11n',
        'Edge', 'Edge', 'Edge'
    ],
    'PointName': [
        'keyframe=6', 'keyframe=5', 'keyframe=4', 'keyframe=3', 'keyframe=2', 'keyframe=1',
        'keyframe=3', 'keyframe=2', 'keyframe=1',
        '11n', '11s', '11x'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create the Base Plot
g = sns.lmplot(
    data=df,
    x='FPS',
    y='MOTA',
    hue='Group',
    height=7,
    aspect=1.3,
    lowess=True,
    ci=None,
    scatter_kws={"s": 50}
)

# --- Set the Requested Limits ---
g.ax.set_ylim(0, 35)
g.ax.set_xlim(0, 30)

# Add Individual Text Labels
# Note: Since adjustText isn't always available, we use standard text placement
texts = []
for index, row in df.iterrows():
    g.ax.text(
        x=row['FPS'],
        y=row['MOTA'],
        s=row['PointName'],
        fontsize=9
    )

# Customize
g.ax.set_title('MOTA vs. FPS for Different Models and Settings', fontsize=16)
g.ax.set_xlabel('FPS (Frames Per Second)', fontsize=12)
g.ax.set_ylabel('MOTA (Multiple Object Tracking Accuracy)', fontsize=12)
g.ax.grid(True, linestyle='--', alpha=0.6)

# Save or show
# plt.show()
plt.savefig('mota_vs_fps.png', bbox_inches='tight')