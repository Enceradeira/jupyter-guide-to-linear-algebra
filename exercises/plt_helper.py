import matplotlib.pyplot as plt
import subprocess
import uuid


def open_plt():
    plot_path = f"/tmp/{uuid.uuid4().hex}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plt.close()
    subprocess.run(['xdg-open', plot_path])
