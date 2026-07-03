from pathlib import Path
import matplotlib.pyplot as plt
from config import OUTPUT_DIR


class BaseVisualizer:

    def __init__(self, save_dir= OUTPUT_DIR / "figures"):

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig, filename):

        save_path = self.save_dir / filename

        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

        print(f"[Saved] {save_path}")