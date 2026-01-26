import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from collections import defaultdict

from .utils import wrap_and_count
from .schema import VariableSpec

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
        percent=False,
        fix_yaxis=False,
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

        # group by filter variable
        filters_by_var = defaultdict(list)

        for filt, df_filtered in filtered_dfs:
            if isinstance(filt.variable, list):
                tmp_variable = set(filt.variable)
                key = "|".join(sorted(tmp_variable))
            else:
                key = filt.variable
            filters_by_var[key].append((filt, df_filtered))

        n_rows = len(variables)
        n_cols = len(filters_by_var.keys())

        # --------------------------------------------------------------
        # Determine max y axis (for fix_yaxis)
        # --------------------------------------------------------------
        y_max = None
        if fix_yaxis:
            if percent:
                y_max = 100
            else:
                y_max = 0
                for _, df_filtered in filtered_dfs:
                    for var in variables:
                        if hasattr(var, "kind"):
                            if var.kind == "singlechoice":
                                y_max = max(y_max, df_filtered[var.name].count())
                            elif var.kind == "singlechoice_text":
                                y_max = max(y_max, df_filtered[var.name].count())
                            elif var.kind == "multichoice":
                                for col in var.columns:
                                    y_max = max(y_max, (df_filtered[col].astype(float) == 2).sum())
                        else:
                            y_max = max(y_max, len(df_filtered))

        # --------------------------------------------------------------
        # Estimate row heights
        # --------------------------------------------------------------
        row_heights = []
        if orient == "h":
            for var in variables:
                max_height = 0
                for _, df_filtered in filtered_dfs:
                    h = self.estimate_row_height(df_filtered, var)
                    max_height = max(max_height, h)
                row_heights.append(max_height)
        else:
            row_heights = [figsize_per_plot[1] / 2] * n_rows
            
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


        kwargs_non_countplot = kwargs.copy()
        kwargs_countplot = kwargs.copy()
        if "color" in kwargs_non_countplot and "palette" in kwargs_non_countplot:
            kwargs_non_countplot.pop("palette", None)
            kwargs_countplot.pop("color", None)
        elif "palette" in kwargs_non_countplot and not "color" in kwargs_non_countplot:
            kwargs_non_countplot["color"] = kwargs_non_countplot.pop("palette", None)[0]

        # --------------------------------------------------------------
        # Plot loop
        # --------------------------------------------------------------
        for j, key in enumerate(filters_by_var.keys()):
            filt, df_filtered = filters_by_var[key][0]
            filter_name = self._resolve_filter_name(filt)

            for i, var in enumerate(variables):
                ax = axes[i, j]
                self._configure_axis(ax, is_left=(j == 0))

                # if filters share the same variable, do grouped plot
                if isinstance(var, VariableSpec) and len(filters_by_var[key]) > 1:
                    
                    self._grouped_countplot(ax, df_filtered, var, filters_by_var[key], orient=orient, percent=percent, **kwargs_countplot)
                    if isinstance(filt.variable, list):
                        filter_names = [self.schema.specs[v].long_name if v in self.schema.specs else self.schema.specs[v.split("_")[0]].long_name for v in list(set(filt.variable))]
                        sub_title = f"Filter: " + f" {filt.mode} ".join(filter_names)
                    else:
                        filter_name = self.schema.specs[filt.variable].long_name if filt.variable in self.schema.specs else self.schema.specs[filt.variable.split("_")[0]].long_name
                        sub_title = f"Filter: {filter_name}"
                    ax.set_title(sub_title)

                else:
                    if hasattr(var, "kind"):
                        if var.kind == "singlechoice":
                            self._plot_singlechoice(ax, df_filtered, var, orient, percent=percent, **kwargs_non_countplot)
                            title_var = var.long_name

                        elif var.kind == "singlechoice_text":
                            self._plot_singlechoice_text(
                                ax, df_filtered, var, orient,
                                mode=singlechoice_text_mode,
                                percent=percent,
                                **kwargs_non_countplot
                            )
                            title_var = var.long_name

                        elif var.kind == "multichoice":
                            self._plot_multichoice(ax, df_filtered, var, orient, percent=percent, **kwargs_non_countplot)
                            title_var = self.schema.short_label(var.name)
                        else:
                            continue
                    else:
                        self._plot_string_variable(ax, df_filtered, var, orient, **kwargs_non_countplot)
                        title_var = var
                        try:
                            title_var = self.schema.short_label(var)
                        except Exception:
                            pass
                    ax.set_title(f"{title_var}  N={len(df_filtered)} ({filter_name})")

                # rotate labels
                self._rotate_category_labels(ax, orient, rotate)

                # fix y axis
                if fix_yaxis and y_max is not None:
                    if orient == "v":
                        ax.set_ylim(0, y_max * 1.05)
                    else:
                        ax.set_xlim(0, y_max * 1.05)


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

    def heatmap_summary(self, filtered_dfs, variables, percent=False, path=None):
        rows = []

        for filt, df_filtered in filtered_dfs:
            filter_name = self._resolve_filter_name(filt)
            total = len(df_filtered)

            for var in variables:

                # -------- VariableSpec ----------
                if isinstance(var, VariableSpec):

                    if var.kind in ("singlechoice", "singlechoice_text"):
                        count = df_filtered[var.name].notna().sum()

                    elif var.kind == "multichoice":
                        count = 0
                        for col in var.columns:
                            count += (df_filtered[col].astype(float) == 2).sum()

                    else:
                        continue

                    label = var.long_name

                # -------- plain column ----------
                else:
                    count = df_filtered[var].notna().sum()
                    label = var

                value = count / total * 100 if percent else count

                rows.append({
                    "Variable": label,
                    "Filter": filter_name,
                    "Value": value,
                })

        df_heat = pd.DataFrame(rows)

        aggfunc = "mean" if percent else "sum"

        df_heat = df_heat.pivot_table(
            index="Variable",
            columns="Filter",
            values="Value",
            aggfunc=aggfunc
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            df_heat,
            annot=True,
            fmt=".1f" if percent else "d",
            cmap="Blues",
        )
        plt.title("Survey Summary Heatmap")
        plt.tight_layout()
        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_singlechoice(self, ax, df, var, orient, percent=False, **kwargs):
        rows = []
        label = self.schema.short_label(var.name)
        value_map = self.df_variables[var.name].values[1]

        series = pd.to_numeric(df[var.name], errors="coerce")
        total = series.notna().sum()

        for code_str, text in value_map.items():
            try:
                code = int(code_str)
            except ValueError:
                continue

            count = (series == code).sum()
            value = count / total * 100 if percent else count
            rows.append({label: text, "value": value})

        df_plot = pd.DataFrame(rows)

        if df_plot.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            return

        self._barplot(ax, df_plot, label, orient, **kwargs)

    def _plot_multichoice(self, ax, df, var, orient, percent=False, **kwargs):
        rows = []
        label = var.long_name
        total = len(df)

        for col in var.columns:
            option_label = self.schema.short_label(col)
            count = (df[col].astype(float) == 2).sum()
            label_count = "count"
            if percent:
                count = count / total * 100
                label_count = "percent"
            rows.append({label: option_label, label_count: count})

        df_plot = pd.DataFrame(rows)
        self._barplot(ax, df_plot, label, orient, **kwargs)

    def _plot_singlechoice_text(
        self,
        ax,
        df,
        var,
        orient,
        mode="ignore",
        percent=False,
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
            self._plot_singlechoice(ax, df, var, orient, percent=percent, **kwargs)
            return

        if mode != "absorb":
            raise ValueError(
                f"Unknown singlechoice_text_mode '{mode}' "
                "(use 'ignore' or 'absorb')"
            )

        df_local = df.copy()
        value_map = self.df_variables[base].values[1]

        # find next available numeric code
        existing_codes = [int(k) for k in value_map.keys() if str(k).isdigit()]
        next_code = max(existing_codes, default=0) + 1

        # collect unique text answers
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
        temp_value_map = dict(value_map)
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

        self._plot_singlechoice_with_custom_labels(
            ax,
            df_local,
            base,
            temp_value_map,
            orient,
            percent=percent,
            **kwargs,
        )

    def _plot_singlechoice_with_custom_labels(
        self,
        ax,
        df,
        base_var,
        value_map,
        orient,
        percent=False,
        **kwargs,
    ):
        rows = []
        label = self.schema.short_label(base_var)
        total = len(df)

        for code_str, text in value_map.items():
            try:
                code = int(code_str)
            except ValueError:
                continue

            count = (df[base_var] == code).sum()
            if percent:
                count = count / total * 100
            rows.append({label: text, "count": count})

        df_plot = pd.DataFrame(rows)
        self._barplot(ax, df_plot, label, orient, **kwargs)

    def _plot_string_variable(self, ax, df, col, orient, **kwargs):        
        label = df.keys()[1]
        if orient == "h":
            sns.countplot(y=col, data=df, ax=ax, **kwargs)
            ax.set_xlabel(label)
        else:
            sns.countplot(x=col, data=df, ax=ax, **kwargs)
            ax.set_ylabel(label)

    def _barplot(self, ax, df_plot, label, orient, **kwargs):
        axis_label = df_plot.keys()[1]
        if orient == "h":
            sns.barplot(y=label, x=axis_label, data=df_plot, ax=ax, **kwargs)
            ax.set_xlabel(axis_label)
        else:
            sns.barplot(x=label, y=axis_label, data=df_plot, ax=ax, **kwargs)
            ax.set_ylabel(axis_label)

    def _grouped_countplot(
        self,
        ax,
        df,
        var,
        filters,
        orient="v",
        percent=False,
        singlechoice_text_mode="ignore",
        **kwargs,
    ):
        label = self.schema.short_label(var.name)
        all_rows = []

        for filt, df_filtered in filters:
            filter_name = self._resolve_filter_name(filt)
            counts = self._collect_counts(
                df_filtered,
                var,
                percent=percent,
                singlechoice_text_mode=singlechoice_text_mode,
            )

            counts["Filter"] = f"{filter_name} (N={len(df_filtered)})"
            all_rows.append(counts)

        plot_df = pd.concat(all_rows, ignore_index=True)
        if orient == "v":
            sns.barplot(
                x="category",
                y="value",
                hue="Filter",
                data=plot_df,
                ax=ax,
                **kwargs,
            )
            ax.set_ylabel("Percentage" if percent else "Count")
        else:
            sns.barplot(
                y="category",
                x="value",
                hue="Filter",
                data=plot_df,
                ax=ax,
                **kwargs,
            )
            ax.set_xlabel("Percentage" if percent else "Count")

        ax.set_xlabel(label if orient == "v" else "")

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

    def _collect_counts(self, df, var, percent=False, singlechoice_text_mode="ignore"):
        """
        Returns DataFrame with columns:
        - category
        - value (count or percent)
        """
        rows = []
        total = len(df)
        if not hasattr(var, "kind"):
            raise TypeError(
                f"_collect_counts expected VariableSpec, got {type(var)}"
            )
        # ---------------- singlechoice ----------------
        if var.kind == "singlechoice":
            series = pd.to_numeric(df[var.name], errors="coerce")
            value_map = self.df_variables[var.name].values[1]
            for code_str, text in value_map.items():
                try:
                    code = int(code_str)
                except ValueError:
                    continue

                count = (series == code).sum()
                value = count / total * 100 if percent else count
                rows.append({"category": text, "value": value})

        # -------- singlechoice + free text ------------
        elif var.kind == "singlechoice_text":
            if singlechoice_text_mode == "ignore":
                # Treat exactly like singlechoice
                base_spec = self.schema.specs[var.name]
                base_spec.kind = "singlechoice"
                return self._collect_counts(df, base_spec, percent)

            # absorb logic (reuse existing logic)
            df_local = df.copy()
            base = var.name
            text_cols = [c for c in var.columns if c != base]
            value_map = self.df_variables[base].values[1]

            existing_codes = [int(k) for k in value_map if k.isdigit()]
            next_code = max(existing_codes, default=0) + 1

            texts = (
                pd.concat([df_local[c] for c in text_cols])
                .dropna()
                .astype(str)
                .str.strip()
            )
            texts = texts[texts != ""].unique()

            temp_map = dict(value_map)
            for txt in texts:
                temp_map[str(next_code)] = txt
                for col in text_cols:
                    df_local.loc[df_local[col] == txt, base] = next_code
                next_code += 1

            for code_str, text in temp_map.items():
                try:
                    code = int(code_str)
                except ValueError:
                    continue
                count = (df_local[base] == code).sum()
                value = count / total * 100 if percent else count
                rows.append({"category": text, "value": value})

        # ---------------- multichoice ----------------
        elif var.kind == "multichoice":
            for col in var.columns:
                label = self.schema.short_label(col)
                count = (df[col].astype(float) == 2).sum()
                value = count / total * 100 if percent else count
                rows.append({"category": label, "value": value})

        return pd.DataFrame(rows)

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

        # correct number of categories
        if hasattr(var, "kind") and var.kind == "singlechoice":
            n_categories = len(self.df_variables[var.name].values[1])
            longest_label = var.long_name
        elif hasattr(var, "kind") and var.kind == "multichoice":
            n_categories = len(var.columns)
            longest_label = ""
            for col in var.columns:
                label = self.df_variables[col].values[0].split("(")[0]
                longest_label = max(longest_label, label, key=len)
        elif hasattr(var, "kind") and var.kind == "singlechoice_text":
            n_categories = len(self.df_variables[var.name].values[1])
            longest_label = var.long_name
        else:
            n_categories = len(df[col].unique())
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

