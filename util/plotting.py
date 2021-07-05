import matplotlib.pyplot as plt
import os
import math


# copying some util functions for execution in jupyter notebooks,
# to avoid import and path issues


def calculate_nrows_ncols(latent_dim_n):
    # calculate number of subplots on canvas
    nrows = int(round(math.sqrt(latent_dim_n/2)))
    ncols = math.ceil(math.sqrt(latent_dim_n/2))
    return nrows, ncols


def plot_clusters(latent_coords, cluster_assignments, labels=['BG', 'SIG'], cluster_centers=None, title_suffix=None, filename_suffix=None, fig_dir=None):

    latent_dim_n = latent_coords.shape[1] - 1 if latent_coords.shape[1] % 2 else latent_coords.shape[1] # if num latent dims is odd, slice off last dim
    nrows, ncols = calculate_nrows_ncols(latent_dim_n)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)

    for d, ax in zip(range(0, latent_dim_n, 2), axs.flat if latent_dim_n > 2 else [axs]):
        scatter = ax.scatter(latent_coords[:,d], latent_coords[:,d+1], c=cluster_assignments, s=100, marker="o", cmap='Dark2')
        ax.set_title(r'$z_{} \quad & \quad z_{}$'.format(d+1, d+2), fontsize='small')
        if cluster_centers is not None:
            ax.scatter(cluster_centers[:, d], cluster_centers[:, d+1], c='black', s=100, alpha=0.5);

    if latent_dim_n > 2 and axs.size > latent_dim_n/2:
        for a in axs.flat[int(latent_dim_n/2):]: a.axis('off')

    legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Classes")
    ax.add_artist(legend1)

    plt.suptitle(' '.join(filter(None, ['data', title_suffix])))
    plt.tight_layout()
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, '_'.join(filter(None, ['clustering', filename_suffix, '.png']))))
    else:
        plt.show()
    plt.close(fig)
