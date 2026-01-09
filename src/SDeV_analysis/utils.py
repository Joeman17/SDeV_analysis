import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import textwrap



def wrap_and_count(text, width):
    wrapped = textwrap.fill(text, width=width)
    n_lines = wrapped.count("\n") + 1
    return wrapped, n_lines


def dataframe_to_pdf(
    df,
    filename,
    title="",
    header_color="#40466e",
    row_colors=("#f1f1f2", "white"),
    edge_color="black",
    font_size=10,
    max_col_width=35,
    base_row_height=1.2,
    left_margin=0.05,
    right_margin=0.05,
    top_margin=0.06,
    bottom_margin=0.06,
):
    """
    Save a styled DataFrame to a single-page PDF with:
    - adaptive column widths
    - adaptive row heights
    - configurable left/right/top/bottom margins
    - title always above the table
    """

    # ---------- Wrap text ----------
    def wrap_text(text, width):
        return "\n".join(textwrap.wrap(str(text), width=width))

    df_wrapped = df.copy()
    for col in df.columns:
        df_wrapped[col] = df[col].apply(lambda x: wrap_text(x, max_col_width))

    # ---------- Compute column widths ----------
    col_max_chars = []
    for col in df_wrapped.columns:
        max_len = max(
            df_wrapped[col]
            .astype(str)
            .map(lambda x: max(len(line) for line in x.split("\n")))
            .max(),
            len(str(col))
        )
        col_max_chars.append(max_len)

    col_widths = np.array(col_max_chars, dtype=float)
    col_widths /= col_widths.sum()  # normalized

    # ---------- Compute row heights ----------
    row_heights = []
    for i in range(len(df_wrapped)):
        max_lines = max(
            df_wrapped.iloc[i]
            .astype(str)
            .map(lambda x: x.count("\n") + 1)
        )
        row_heights.append(base_row_height * max_lines)

    # ---------- Figure layout ----------
    fig = plt.figure(figsize=(11.7, 8.3))  # A4 landscape

    # Axes covering only the area inside left/right margins
    ax = fig.add_axes([left_margin, bottom_margin, 1 - left_margin - right_margin, 1 - top_margin - bottom_margin])
    ax.axis("off")

    # Scale column widths to Axes width
    col_widths_scaled = col_widths * ax.get_position().width

    # ---------- Create table ----------
    table = ax.table(
        cellText=df_wrapped.values,
        colLabels=df_wrapped.columns,
        colWidths=col_widths_scaled,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # ---------- Header styling ----------
    for col in range(len(df.columns)):
        cell = table[(0, col)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor(edge_color)

    # ---------- Body styling & row heights ----------
    total_row_height = sum(row_heights)
    for row in range(1, len(df) + 1):
        for col in range(len(df.columns)):
            cell = table[(row, col)]
            cell.set_facecolor(row_colors[row % 2])
            cell.set_edgecolor(edge_color)
            # scale row height relative to total
            cell.set_height(row_heights[row - 1] / total_row_height)

    # ---------- Shift table vertically to respect top/bottom margins ----------
    # Compute total table height in figure fraction
    table_height = sum(row_heights) / total_row_height
    # Shift so bottom margin is preserved
    for key, cell in table.get_celld().items():
        cell.set_y(bottom_margin + (cell.get_y() - 0.0) * (1 - top_margin - bottom_margin))

    # ---------- Title ----------
    if title:
        fig.text(
            0.5,
            1 - top_margin / 2,
            title,
            ha="center",
            va="center",
            fontsize=14,
            weight="bold"
        )

    # ---------- Save PDF ----------
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches="tight")

    plt.close(fig)