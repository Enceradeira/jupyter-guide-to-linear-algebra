import matplotlib.pyplot as plt
import subprocess

def open_plt():
    # Print plot
    plot_path = '/home/jorg/projects/jj/jupyter-guide-to-linear-algebra/plt.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plt.close()
    subprocess.run(['xdg-open', plot_path])
