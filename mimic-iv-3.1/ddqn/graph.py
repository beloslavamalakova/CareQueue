import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\20243009\CareQueue\mimic-iv-3.1\ddqn\outputs\metrics.csv")

for col in ['val_bc', 'val_q']:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(df['epoch'], df['val_bc'], marker='o')
ax1.set_title('Normalised Validation Behaviour Cloning Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Normalised Loss [0,1]')
ax1.set_ylim(0, 1)

ax2.plot(df['epoch'], df['val_q'], marker='o')
ax2.set_title('Normalised Validation Q Bellman Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Normalised Loss [0,1]')
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('plots.png', dpi=150)
plt.show()
