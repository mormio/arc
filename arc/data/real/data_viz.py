import matplotlib.pyplot as plt
from matplotlib import colors

cmap = colors.ListedColormap(
    [
        "#000000",
        "#0074D9",
        "#FF4136",
        "#2ECC40",
        "#FFDC00",
        "#AAAAAA",
        "#F012BE",
        "#FF851B",
        "#7FDBFF",
        "#870C25",
    ]
)
norm = colors.Normalize(vmin=0, vmax=9)


def plot_task(task, task_solutions, i, t):
    """Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app"""

    num_train = len(task["train"])
    num_test = len(task["test"])

    w = num_train + num_test
    fig, axs = plt.subplots(2, w, figsize=(3 * w, 3 * 2))
    plt.suptitle(f"Set #{i}, {t}:", fontsize=20, fontweight="bold", y=1)
    # plt.subplots_adjust(hspace = 0.15)
    # plt.subplots_adjust(wspace=20, hspace=20)

    for j in range(num_train):
        plot_one(task, axs[0, j], j, "train", "input")
        plot_one(task, axs[1, j], j, "train", "output")

    plot_one(task, axs[0, j + 1], 0, "test", "input")

    answer = task_solutions
    input_matrix = answer

    axs[1, j + 1].imshow(input_matrix, cmap=cmap, norm=norm)
    axs[1, j + 1].grid(True, which="both", color="lightgrey", linewidth=0.5)
    axs[1, j + 1].set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    axs[1, j + 1].set_xticks(
        [x - 0.5 for x in range(1 + len(input_matrix[0]))]
    )
    axs[1, j + 1].set_xticklabels([])
    axs[1, j + 1].set_yticklabels([])
    axs[1, j + 1].set_title("Test output")

    axs[1, j + 1] = plt.figure(1).add_subplot(111)
    axs[1, j + 1].set_xlim([0, num_train + 1])

    for m in range(1, num_train):
        axs[1, j + 1].plot([m, m], [0, 1], "--", linewidth=1, color="black")

    axs[1, j + 1].plot(
        [num_train, num_train], [0, 1], "-", linewidth=3, color="black"
    )

    axs[1, j + 1].axis("off")

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor("black")
    fig.patch.set_facecolor("#dddddd")

    plt.tight_layout()

    print(f"#{i}, {t}")  # for fast and convinience search
    plt.show()

    print()
    print()


def plot_one(task, ax, i, train_or_test, input_or_output):
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)

    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])

    ax.set_title(train_or_test + " " + input_or_output)
