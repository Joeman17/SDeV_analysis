import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from .utils import wrap_and_count

class SurveyPlotter:
    def __init__(self, schema):
        """
        schema: SurveySchema
        """
        self.schema = schema
        self.df_variables = schema.df_variables

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def countplot_grid(
        self,
        filtered_dfs,
        variables,
        figsize_per_plot=(5, 4),
        orient="v",
        title=None,
        wrap_width=20,
        rotate=0,
        path=None,
        singlechoice_text_mode="ignore",
        **kwargs
    ):
        """
        Plot a grid of countplots:
        - rows = variables
        - columns = filters

        filtered_dfs: list[(Filter, DataFrame)]
        variables: list[str | VariableSpec]
        """

        # allow overrides via kwargs (backward compatible)
        if "rotate" in kwargs:
            rotate = kwargs.pop("rotate")
        if "wrap_width" in kwargs:
            wrap_width = kwargs.pop("wrap_width")

        # --------------------------------------------------------------
        # Estimate row heights
        # --------------------------------------------------------------

        row_heights = []
        for var in variables:
            max_height = 0
            for _, df_filtered in filtered_dfs:
                h = self.estimate_row_height(df_filtered, var)
                max_height = max(max_height, h)
            row_heights.append(max_height)

        n_rows = len(variables)
        n_cols = len(filtered_dfs)

        fig_width = figsize_per_plot[0] * n_cols
        fig_height = figsize_per_plot[1] * sum(row_heights)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_height),
            gridspec_kw={"height_ratios": row_heights},
        )

        # normalize axes to 2D
        axes = self._ensure_2d_axes(axes, n_rows, n_cols)

        # --------------------------------------------------------------
        # Plot loop
        # --------------------------------------------------------------

        for j, (filt, df_filtered) in enumerate(filtered_dfs):

            filter_name = self._resolve_filter_name(filt)

            for i, var in enumerate(variables):
                ax = axes[i, j]

                self._configure_axis(ax, is_left=(j == 0))

                if hasattr(var, "kind"):
                    if var.kind == "singlechoice":
                        self._plot_singlechoice(ax, df_filtered, var, orient, **kwargs)
                        title_var = var.long_name
                        
                    elif var.kind == "singlechoice_text":
                        mode = kwargs.pop("singlechoice_text_mode", "ignore")
                        self._plot_singlechoice_text(
                            ax, df_filtered, var, orient, mode=mode, **kwargs
                        )
                        title_var = var.long_name

                    elif var.kind == "multichoice":
                        self._plot_multichoice(ax, df_filtered, var, orient, **kwargs)
                        title_var = var.name
                    else:
                        continue
                else:
                    self._plot_string_variable(ax, df_filtered, var, orient, **kwargs)
                    title_var = var
                self._rotate_category_labels(ax, orient, rotate)
                ax.set_title(f"{title_var}  N={len(df_filtered)} ({filter_name})")

                if j == 0:
                    max_lines = wrap_yticks(ax, width=wrap_width)
                    base_hspace = 0.25
                    line_increment = 0.12
                    hspace = base_hspace + (max_lines - 1) * line_increment
        
        fig.subplots_adjust(hspace=hspace)

        if title:
            fig.suptitle(title, fontsize=16, weight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_singlechoice(self, ax, df, var, orient, **kwargs):
        rows = []

        label = self.schema.short_label(var.name)
        value_map = self.df_variables[var.name].values[1]

        for code_str, text in value_map.items():
            try:
                code = int(code_str)
            except ValueError:
                continue

            count = (df[var.name] == code).sum()
            rows.append({label: text, "count": count})

        df_plot = pd.DataFrame(rows)

        self._barplot(ax, df_plot, label, orient, **kwargs)

    def _plot_multichoice(self, ax, df, var, orient, **kwargs):
        rows = []

        label = var.long_name

        for col in var.columns:
            option_label = self.schema.short_label(col)

            # explicit numeric comparison
            count = (df[col].astype(float) == 2).sum()

            rows.append({label: option_label, "count": count})

        df_plot = pd.DataFrame(rows)

        self._barplot(ax, df_plot, label, orient, **kwargs)

    def _plot_singlechoice_text(
        self,
        ax,
        df,
        var,
        orient,
        mode="ignore",
        **kwargs,
    ):
        """
        Plot singlechoice_text variables.

        mode:
        - "ignore": plot like singlechoice, ignore text columns
        - "absorb": convert unique text answers into categories
        """

        base = var.name
        text_cols = [c for c in var.columns if c != base]

        if mode == "ignore":
            # behave exactly like singlechoice
            self._plot_singlechoice(ax, df, var, orient, **kwargs)
            return

        if mode != "absorb":
            raise ValueError(
                f"Unknown singlechoice_text_mode '{mode}' "
                "(use 'ignore' or 'absorb')"
            )

        # --------------------------------------------------
        # ABSORB MODE (plot-time only)
        # --------------------------------------------------

        df_local = df.copy()

        value_map = self.df_variables[base].values[1]

        # find next available numeric code
        existing_codes = [
            int(k) for k in value_map.keys() if str(k).isdigit()
        ]
        next_code = max(existing_codes, default=0) + 1

        # collect unique text answers across ALL text columns
        texts = []
        for col in text_cols:
            texts.append(df_local[col])

        texts = (
            pd.concat(texts, axis=0)
            .dropna()
            .astype(str)
            .str.strip()
        )
        texts = texts[texts != ""]

        unique_texts = sorted(texts.unique())

        # build temporary label map
        temp_value_map = dict(value_map)  # copy, do NOT mutate schema
        text_to_code = {}

        for txt in unique_texts:
            code = next_code
            next_code += 1
            temp_value_map[str(code)] = txt
            text_to_code[txt] = code

        # inject codes into base variable
        for col in text_cols:
            for txt, code in text_to_code.items():
                mask = df_local[col].astype(str).str.strip() == txt
                df_local.loc[mask, base] = code

        # --------------------------------------------------
        # plot using normal singlechoice logic
        # --------------------------------------------------

        self._plot_singlechoice_with_custom_labels(
            ax,
            df_local,
            base,
            temp_value_map,
            orient,
            **kwargs,
        )
    def _plot_singlechoice_with_custom_labels(
        self,
        ax,
        df,
        base_var,
        value_map,
        orient,
        **kwargs,
    ):
        rows = []

        label = self.schema.short_label(base_var)

        for code_str, text in value_map.items():
            try:
                code = int(code_str)
            except ValueError:
                continue

            count = (df[base_var] == code).sum()
            rows.append({label: text, "count": count})

        df_plot = pd.DataFrame(rows)

        self._barplot(ax, df_plot, label, orient, **kwargs)

    def _plot_string_variable(self, ax, df, col, orient, **kwargs):
        if orient == "h":
            sns.countplot(y=col, data=df, ax=ax, **kwargs)
            ax.set_xlabel("Count")
        else:
            sns.countplot(x=col, data=df, ax=ax, **kwargs)
            ax.set_ylabel("Count")

    def _barplot(self, ax, df_plot, label, orient, **kwargs):
        if orient == "h":
            sns.barplot(y=label, x="count", data=df_plot, ax=ax, **kwargs)
            ax.set_xlabel("Count")
        else:
            sns.barplot(x=label, y="count", data=df_plot, ax=ax, **kwargs)
            ax.set_ylabel("Count")

    # ------------------------------------------------------------------
    # Axis & layout helpers
    # ------------------------------------------------------------------

    def _configure_axis(self, ax, is_left):
        if not is_left:
            ax.tick_params(left=False)
            ax.get_yaxis().set_visible(False)
        else:
            ax.tick_params(left=True)
            ax.get_yaxis().set_visible(True)

    def _ensure_2d_axes(self, axes, n_rows, n_cols):
        if n_rows == 1 and n_cols == 1:
            return np.array([[axes]])
        if n_rows == 1:
            return axes[np.newaxis, :]
        if n_cols == 1:
            return axes[:, np.newaxis]
        return axes

    def _resolve_filter_name(self, filt):
        if filt.name is not None:
            return filt.name

        # fallback: use value label
        if isinstance(filt.variable, list):
            return " & ".join(filt.variable)

        try:
            return self.df_variables[filt.variable].values[1][str(filt.condition)]
        except Exception:
            return str(filt.condition)
        
    def _rotate_category_labels(self, ax, orient, rotate):
        """
        Rotate category tick labels depending on plot orientation.

        orient="v": categories on x-axis
        orient="h": categories on y-axis
        """
        if rotate == 0:
            return

        if orient == "v":
            ax.tick_params(axis="x", labelrotation=rotate)
            for label in ax.get_xticklabels():
                label.set_ha("right" if rotate > 0 else "center")

        elif orient == "h":
            ax.tick_params(axis="y", labelrotation=rotate)
            for label in ax.get_yticklabels():
                label.set_va("center")

    # ------------------------------------------------------------------
    # Height estimation (unchanged logic)
    # ------------------------------------------------------------------

    def estimate_row_height(
        self,
        df,
        var,
        wrap_width=35,
        base_per_category=0.35,
        line_increment=0.12,
        min_height=1.2,
    ):
        import textwrap

        n_categories = len(df.columns)

        if hasattr(var, "kind") and var.kind == "singlechoice":
            longest_label = var.long_name

        elif hasattr(var, "kind") and var.kind == "multichoice":
            longest_label = ""
            for col in var.columns:
                label = self.df_variables[col].values[0].split("(")[0]
                longest_label = max(longest_label, label, key=len)
        else:
            longest_label = str(var)

        wrapped = textwrap.fill(longest_label, width=wrap_width)
        n_lines = wrapped.count("\n") + 1

        height = (
            n_categories * base_per_category
            + (n_lines - 1) * line_increment
        )

        return max(height, min_height)

def plot_barplot(data, x, y, title="Bar Plot", xlabel="X-axis", ylabel="Y-axis"):
    """
    Plots a bar plot using seaborn.

    Parameters:
    - data: DataFrame containing the data to plot
    - x: Column name for x-axis
    - y: Column name for y-axis
    - title: Title of the plot
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x=x, y=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()



def plot_countplot(df, x, title="Count Plot", xlabel="X-axis", ylabel="Y-axis", **kwargs):
    """
    Plots a count plot using seaborn.

    Parameters:
    - df: DataFrame containing the data to plot
    - x: Column name for x-axis
    - title: Title of the plot
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=x, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def wrap_yticks(ax, width=35):
    row_labels = [
        t.get_text()
        for t in ax.get_yticklabels()
    ]

    wrapped_labels = []
    line_counts = []

    for lbl in row_labels:
        wrapped, n = wrap_and_count(lbl, width)
        wrapped_labels.append(wrapped)
        line_counts.append(n)

    max_lines = max(line_counts)
    ax.set_yticklabels(wrapped_labels)
    return max_lines

