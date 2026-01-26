import operator
import json
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

from .visualize import plot_countplot, wrap_yticks
from .schema import SurveySchema, VariableSpec


OP_MAP = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "in": lambda s, v: s.isin(v),
    "not in": lambda s, v: ~s.isin(v),
}

class Filter(object):
    def __init__(self, variable, condition_func, condition, mode="or", name=None):
        """
        Docstring for __init__
        
        :param self: Description
        :param variable: Description
        :param condition_func: Description
        :param condition: Description
        :param name: Description
        """
        self.condition = condition
        self.variable = variable
        self.condition_func = self.handle_condition_func(condition_func)
        self.mode = mode
        self.name = name

        if self.mode not in ("and", "or"):
            raise ValueError("mode must be 'and' or 'or'")

    def handle_condition_func(self, condition_func):
        if isinstance(condition_func, str):
            return OP_MAP[condition_func]
        elif isinstance(condition_func, list):
            for i in range(len(condition_func)):
                condition_func[i] = self.handle_condition_func(condition_func[i])
            return condition_func
        else:
            return condition_func

    def apply(self, df):
        # ----------------------------------------
        # MULTI-CONDITION
        # ----------------------------------------
        if isinstance(self.condition, list):
            if not (len(self.condition) == len(self.condition_func) == len(self.variable)):
                raise ValueError(
                    "Length of condition, condition_func, and variable lists must be the same."
                )

            if self.mode == "and":
                result = pd.Series(True, index=df.index)
                for var, cond_func, cond in zip(self.variable, self.condition_func, self.condition):
                    result &= cond_func(df[var], cond)

            else:  # OR
                result = pd.Series(False, index=df.index)
                for var, cond_func, cond in zip(self.variable, self.condition_func, self.condition):
                    result |= cond_func(df[var], cond)

            return result

        # ----------------------------------------
        # SINGLE CONDITION
        # ----------------------------------------
        return self.condition_func(df[self.variable], self.condition)

class SurveyAnalysis:
    def __init__(self, client, schema):
        """
        client: SoSciClient
        schema: SurveySchema
        """
        self.client = client
        self.schema = schema
        self.df_variables = schema.df_variables

    # ------------------------------------------------------------------
    # Core data access
    # ------------------------------------------------------------------

    def get_variables(self, variables, filters):
        """
        Fetch variables and return a list of (Filter, DataFrame) tuples.
        """

        vlist = self._collect_required_columns(variables, filters)
        df = self.client.load_dataframe(vlist)

        result = []
        resolved_cols = self.schema.resolve_columns(variables)

        for filt in filters:
            mask = filt.apply(df)
            df_filtered = df.loc[mask, resolved_cols].copy()
            df_filtered.reset_index(drop=True, inplace=True)
            result.append((filt, df_filtered))

        return result

    # ------------------------------------------------------------------
    # Multichoice extraction
    # ------------------------------------------------------------------

    def extract_multichoice(self, df, variable_spec, picked_value=1):
        """
        Converts multichoice columns into long-format counts.
        """
        rows = []

        for col in variable_spec.columns:
            label = col.replace(variable_spec.name + "_", "")
            count = (df[col] == picked_value).sum()
            rows.append({"option": label, "count": count})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Response rate logic (unchanged, but isolated)
    # ------------------------------------------------------------------

    def get_link_response_rate(self):
        params = "&vSkip="
        keys = list(self.df_variables.keys())[6:-5]

        params += ",".join(keys)

        raw = self.client.send_json(params=params).json()["data"]
        df = pd.DataFrame(raw).transpose()

        df_response_rate = pd.concat(
            [
                df["REF"].value_counts(),
                df[df["MAXPAGE"] > 30]["REF"].value_counts(),
            ],
            axis=1,
            keys=["Alle", "Vollständig ausgefüllt"],
        ).fillna(0).astype(int)

        df_response_rate = pd.merge(
            df_response_rate,
            df[df["REF"].isin(
                df[df["MAXPAGE"] > 30]["REF"].value_counts().index
            )]
            .groupby("REF", as_index=False)["STARTED"]
            .min()
            .set_index("REF"),
            on="REF",
            how="left",
        ).rename(columns={"STARTED": "Veröffentlicht"})

        df_response_rate["Veröffentlicht"] = pd.to_datetime(
            df_response_rate["Veröffentlicht"], errors="ignore"
        ).dt.date

        df_response_rate = df_response_rate.replace({pd.NaT: "Nein"})

        return df_response_rate

    def update_response_rate_sheet(self, path_link_rueckverfolgung):
        link_pd = pd.read_csv(path_link_rueckverfolgung)
        response_rate_pd = (
            self.get_link_response_rate()
            .reset_index()
            .rename(columns={"REF": "Ref"})
        )
        # Normalize "Ref" columns for merging
        for df in (link_pd, response_rate_pd):
            df["Ref"] = (
                df["Ref"]
                .astype(str)
                .str.strip()
                .str.lower()
            )
        merged_pd = pd.merge(
            link_pd, response_rate_pd, on="Ref", how="left"
        )

        merged_pd = (
            merged_pd
            .drop(
                [
                    "Link",
                    "Hyperlink",
                    "Endung",
                    "Aufrufe",
                    "Rückläufe",
                    "Datum d. Veröffentlichung",
                ],
                axis=1,
            )
            .rename(
                columns={
                    "Vollständig ausgefüllt": "Rückläufe",
                    "Alle": "Aufrufe",
                }
            )
        )

        merged_pd = merged_pd.iloc[:, [0, 1, 5, 3, 4, 2]]
        merged_pd = merged_pd.replace({np.nan: ""})

        merged_pd.loc[merged_pd["Aufrufe"] == "", "Aufrufe"] = 0
        merged_pd.loc[merged_pd["Rückläufe"] == "", "Rückläufe"] = 0
        merged_pd = merged_pd.astype({"Aufrufe": "int64", "Rückläufe": "int64"})
        merged_pd.loc['Total', "Aufrufe"] = merged_pd[merged_pd.index != "Total"]["Aufrufe"].astype("int64").sum()
        merged_pd.loc['Total', "Rückläufe"] = merged_pd[merged_pd.index != "Total"]["Rückläufe"].astype("int64").sum()

        return merged_pd
    
    def _collect_required_columns(self, variables, filters=None):
        """
        Resolve all dataframe columns required for variables + filters,
        including numeric-suffix free-text columns (e.g. Axyz_1, Axyz_2).
        """
        cols = set()

        # We need access to *all* available columns
        all_columns = set(self.schema.df_variables.keys())

        # --------------------
        # variables
        # --------------------
        for var in variables:
            if hasattr(var, "kind"):  # VariableSpec
                if var.kind == "multichoice":
                    cols.update(var.columns)
                else:
                    cols.add(var.name)
                    cols.update(
                        self._expand_text_suffix_columns(var.name, all_columns)
                    )
            else:
                cols.add(var)
                cols.update(
                    self._expand_text_suffix_columns(var, all_columns)
                )

        # --------------------
        # filters
        # --------------------
        if filters:
            for filt in filters:
                if isinstance(filt.variable, list):
                    cols.update(filt.variable)
                else:
                    cols.add(filt.variable)

        return cols
    
    def _expand_text_suffix_columns(self, base_var, all_columns):
        """
        Find columns like Axyz_1, Axyz_2, ... for a base variable Axyz.
        """
        pattern = re.compile(rf"^{re.escape(base_var)}_[0-9]+$")
        return {c for c in all_columns if pattern.match(c)}

# class Analysis:
#     def __init__(self, api_link: str):
#         self.api_link = api_link
#         self.df_variables = self._load_variables()
#         self.specs = self.init_specs()
#         self.beautify_variable_labels()

#     def _load_variables(self):
#         data = self.send_json(None, api_link=self.api_link + "&cases=none&infoValues")
#         data_json = data.json()
#         return pd.DataFrame(data_json["variables"])
    
#     def init_specs(self):
#         singlechoice_pattern = re.compile(r"A\d+$")
#         multichoice_pattern = re.compile(r"\d+_\d+$")

#         singlechoice_cols = [
#             col for col in self.df_variables.keys()
#             if singlechoice_pattern.search(col)
#         ]

#         groups = defaultdict(list)

#         for col in self.df_variables.keys():
#             if multichoice_pattern.search(col):
#                 base = col.rsplit("_", 1)[0]
#                 if self.df_variables.keys().__contains__(base):
#                     groups[base].append(col)
#                     if base in singlechoice_cols:
#                         singlechoice_cols.remove(base)

#         specs = dict()
#         for base, cols in groups.items():
#             specs[base] = VariableSpec(name=base, kind="multichoice", columns=cols)

#         for col in singlechoice_cols:
#             specs[col] = VariableSpec(name=col, kind="singlechoice")
#         return specs

    
#     def beautify_variable_labels(self):
#         for col in self.specs.keys():
#             if self.specs[col].kind == "multichoice":
#                 self.df_variables[col].label = self.df_variables[col].label.split(":")[0]
    
#                 for c in self.specs[col].columns:
#                     self.df_variables[c].label = self.df_variables[c].label.split(":")[1]

            

#     def send_json(self, data, api_link=None, apikey=None):
#         if api_link is None:
#             api_link = self.api_link
#         data_json = json.dumps(data)
#         payload = {'json_payload': data_json}
#         if apikey is not None:
#             payload['apikey'] = apikey
#         return requests.get(api_link, data=payload)

#     def get_variable(self, variable: str, filter: Filter):
#         data = self.send_json(None, api_link=self.api_link + "&vList=" + variable + "," + filter.variable)
#         #return data
#         data_json = data.json()
#         filtered = {
#         k: v
#         for k, v in data_json["data"].items()
#         if isinstance(v, dict) and len(v) > 1
#     }
#         df = pd.DataFrame(filtered).T
#         df = df[filter.apply(df)]
#         df = pd.DataFrame(df[variable])
#         return df

#     def get_variables(self, variables, filters):
#         """
#         Fetch variables and return a list of (Filter, DataFrame) tuples.

#         - variables: list of variable names to fetch
#         - filters: list of Filter objects

#         Returns:
#         - list of tuples: [(Filter, DataFrame), ...]
#         """
    
#         # Collect all required columns
#         vlist = set()
#         for var in variables:
#             if isinstance(var, VariableSpec):
#                 if var.kind == "multichoice":
#                     vlist.update(var.columns)
#                 else:
#                     vlist.add(var.name)
#             else:
#                 vlist.add(var)

#         for filt in filters:
#             if isinstance(filt.variable, list):
#                 vlist.update(filt.variable)
#             else:
#                 vlist.add(filt.variable)

#         # 2. Fetch all variables at once
#         api_vlist = ",".join(vlist)
#         data = self.send_json(None, api_link=self.api_link + "&vList=" + api_vlist)
#         data_json = data.json()

#         # 3. Filter out empty/trivial entries
#         filtered_data = {
#             k: v
#             for k, v in data_json["data"].items()
#             if isinstance(v, dict) and len(v) > 1
#         }
#         df = pd.DataFrame(filtered_data).T  # rows = observations

#         for var in variables:
#             # SINGLE variable
#             if isinstance(var, VariableSpec) and var.kind == "singlechoice":
#                 col = var.name

#                 if col in df.columns:
#                     df[col] = df[col].astype("Int64").astype(str)#.replace(self.df_variables[col].values[1])
#                 else:
#                     raise ValueError(f"Column {col} not found in DataFrame.")
#                 # if self.df_variables[col].values[0] is not None:
#                 #     df.rename(columns={col: self.df_variables[col].values[0].split("(")[0]}, inplace=True)

#             # MULTICHOICE variable → do NOTHING here
#             elif isinstance(var, VariableSpec) and var.kind == "multichoice":
#                 # handled later during plotting / aggregation
#                 continue

#             # BACKWARD COMPATIBILITY (plain string)
#             elif isinstance(var, str):
#                 df[var] = df[var].replace(self.df_variables[var].values[1])

#             # df[variable] = df[variable].replace(self.df_variables[variable].values[1])
#             # if self.df_variables[variable].values[0] is not None:
#             #     df.rename(columns={variable: self.df_variables[variable].values[0]}, inplace=True)
#             # else:
#             #     tmp_dict = {}
#             #     for s in self.df_variables.keys():
#             #         if variable + "_" in s:
#             #             tmp_dict[s] = self.df_variables[s].label
#             #     df.rename(columns=tmp_dict, inplace=True)
#         # 4. Apply each filter and return separately
#         result = []
#         resolved_cols = resolve_columns(variables)

#         for filt in filters:
#             mask = filt.apply(df)
#             df_filtered = df.loc[mask, resolved_cols].copy()
#             df_filtered.reset_index(drop=True, inplace=True)
#             result.append((filt, df_filtered))

#         return result

#     def extract_multichoice(self, df, variable_spec, picked_value=1):
#         """
#         Converts multichoice columns into long-format counts.
#         """
#         data = []

#         for col in variable_spec.columns:
#             label = col.replace(variable_spec.name + "_", "")
#             count = (df[col] == picked_value).sum()
#             data.append({"option": label, "count": count})

#         return pd.DataFrame(data)
    
#     def get_link_response_rate(self):
#         params = "&vSkip="
#         for key in self.df_variables.keys()[6:-5]:
#             if params == "&vSkip=":
#                 params += key
#             else:
#                 params += "," + key
#         data = self.send_json(None, api_link=self.api_link + params)
#         data_json = data.json()
#         df = pd.DataFrame(data_json["data"]).transpose()
#         df_response_rate = pd.concat([df["REF"].value_counts(), df[df["MAXPAGE"]>30]["REF"].value_counts()], axis=1, keys=["Alle", "Vollständig ausgefüllt"]).fillna(0).astype(int)
#         #df[df[] == df_response_rate["REF"]]
#         df_response_rate = pd.merge(df_response_rate, df[df["REF"].isin(df[df["MAXPAGE"]>30]["REF"].value_counts().index)].groupby("REF", as_index=False)["STARTED"].min().set_index("REF"), on="REF", how="left").rename(columns={"STARTED": "Veröffentlicht"})
#         df_response_rate['Veröffentlicht'] = pd.to_datetime(df_response_rate['Veröffentlicht'], errors='ignore')
#         df_response_rate["Veröffentlicht"] = df_response_rate["Veröffentlicht"].dt.date
#         df_response_rate = df_response_rate.replace({pd.NaT: "Nein"})
#         return df_response_rate
    
#     def update_response_rate_sheet(self, path_link_rueckverfolgung: str):
#         link_pd = pd.read_csv(path_link_rueckverfolgung)
#         response_rate_pd = self.get_link_response_rate().reset_index().rename(columns={"REF": "Ref"})
#         merged_pd = pd.merge(link_pd, response_rate_pd, on="Ref", how="left")
#         merged_pd = merged_pd.drop('Link', axis=1).drop('Hyperlink', axis=1).drop('Endung', axis=1).drop('Aufrufe', axis=1).drop('Rückläufe', axis=1).drop('Datum d. Veröffentlichung', axis=1)
#         merged_pd = merged_pd.rename(columns={"Vollständig ausgefüllt": "Rückläufe", "Alle": "Aufrufe"})
#         merged_pd = merged_pd.iloc[:, [0,1,5, 3,4,2]]
#         merged_pd = merged_pd.replace({np.nan: ""})
#         return merged_pd
    
#     def barplot(self, variable: str, filter: Filter, plot_settings: dict = {}):
#         df = self.get_variable(variable, filter)
#         df = df.astype(str)
#         df.replace(self.df_variables[variable].values[1], inplace=True)
#         df.rename(columns={variable: self.df_variables[variable].values[0]}, inplace=True)
#         plot_countplot(df, x=self.df_variables[variable].values[0], title=f"{self.df_variables[variable].values[0]} N = {len(df)} ({self.df_variables[filter.variable].values[1][str(filter.condition)]})", xlabel=self.df_variables[variable].values[0], ylabel="Count", **plot_settings)

#     def countplot_grid(self, variables, filters, figsize_per_plot=(5,4), title=None, orient='v', path=None, plot_settings: dict = {}):
#         df_list = self.get_variables(variables, filters)
#         self.plot_grid_countplots(df_list, variables, figsize_per_plot=figsize_per_plot, title=title, orient=orient, path=path, **plot_settings)

#     def plot_grid_countplots(
#         self,
#         filtered_dfs,
#         variables,
#         figsize_per_plot=(5, 4),
#         orient="v",
#         title=None,
#         wrap_width=20,
#         rotate=0,
#         path=None,
#         **kwargs
#     ):
#         """
#         Plot a grid of countplots:
#         - rows = variables
#         - columns = filters (from filtered_dfs)

#         filtered_dfs: list of (Filter, DataFrame) tuples
#         variables: list of str or VariableSpec
#         """

#         if "rotate" in kwargs.keys():
#             rotate = kwargs['rotate'].copy()
#             kwargs['rotate'] = None
        
#         if "wrap_width" in kwargs.keys():
#             wrap_width = kwargs['wrap_width'].copy()
#             kwargs['wrap_width'] = None
        
#         row_heights = []

#         for var in variables:
#             max_height = 0
#             for _, df_filtered in filtered_dfs:
#                 h = self.estimate_row_height(df_filtered, var)
#                 max_height = max(max_height, h)
#             row_heights.append(max_height)
        

#         n_rows = len(variables)
#         n_cols = len(filtered_dfs)

#         fig_width = figsize_per_plot[0] * n_cols
#         fig_height = figsize_per_plot[1] * sum(row_heights)

#         fig, axes = plt.subplots(
#             n_rows,
#             n_cols,
#             figsize=(fig_width, fig_height),
#             gridspec_kw={"height_ratios": row_heights},
#         )

#         # Ensure axes is 2D
#         if n_rows == 1 and n_cols == 1:
#             axes = np.array([[axes]])
#         elif n_rows == 1:
#             axes = axes[np.newaxis, :]
#         elif n_cols == 1:
#             axes = axes[:, np.newaxis]

#         for j, (filt, df_filtered) in enumerate(filtered_dfs):
#             # Resolve filter name
#             if filt.name is not None:
#                 filter_name = filt.name
#             else:
#                 filter_name = self.df_variables[filt.variable].values[1][str(filt.condition)]

#             for i, var in enumerate(variables):
#                 ax = axes[i, j]
#                 if j != 0:
#                     ax.tick_params(left=False)
#                     ax.get_yaxis().set_visible(False)
#                 else:
#                     ax.tick_params(left=True)
#                     ax.get_yaxis().set_visible(True)
#                 # -----------------------
#                 # SINGLE VARIABLE
#                 # -----------------------
#                 if isinstance(var, VariableSpec) and var.kind == "singlechoice":

#                     rows = []

#                     label = var.long_name
                    
#                     for col in self.df_variables[var.name].values[1].keys():
#                         # option_label = self.df_variables[col].values[0].split("(")[0]
#                         count = (df_filtered[var.name] == col).sum()
#                         rows.append({label: self.df_variables[var.name].values[1][col], "count": count})

#                     df_plot = pd.DataFrame(rows)

#                     if orient == "h":
#                         sns.barplot(
#                             y=label,
#                             x="count",
#                             data=df_plot,
#                             ax=ax,
#                             **kwargs,
#                         )
#                         #ax.set_ylabel(var.name)
#                         ax.set_xlabel("Count")
#                     else:
#                         sns.barplot(
#                             x=label,
#                             y="count",
#                             data=df_plot,
#                             ax=ax,
#                             **kwargs,
#                         )
#                         # ax.set_xlabel(var.name)
#                         ax.set_ylabel("Count")
#                     # if var.name in df_filtered.columns:
#                     #     df_filtered = df_filtered.rename(columns={var.name: var.long_name})

#                     # label = var.long_name

#                     # if orient == "h":
#                     #     ylabel = self.df_variables[var.name].values[0]
#                     #     sns.countplot(y=var.long_name, data=df_filtered, ax=ax, **kwargs)
#                     #     ax.set_ylabel(ylabel)
#                     #     ax.set_xlabel("Count")
#                     #     ax.set_yticklabels()
#                     # else:
#                     #     sns.countplot(x=var.long_name, data=df_filtered, ax=ax, **kwargs)
#                     #     # ax.set_xlabel(label)
#                     #     ax.set_ylabel("Count")

#                     ax.set_title(f"{label}  N={len(df_filtered)} ({filter_name})")

#                 # -----------------------
#                 # MULTICHOICE VARIABLE
#                 # -----------------------
#                 elif isinstance(var, VariableSpec) and var.kind == "multichoice":
#                     rows = []

#                     label = var.long_name
                    
#                     for col in var.columns:
#                         option_label = self.df_variables[col].values[0].split("(")[0]
#                         count = (df_filtered[col] == 2).sum()
#                         rows.append({label: option_label, "count": count})

#                     df_plot = pd.DataFrame(rows)



#                     if orient == "h":
#                         sns.barplot(
#                             y=label,
#                             x="count",
#                             data=df_plot,
#                             ax=ax,
#                             **kwargs,
#                         )
#                         #ax.set_ylabel(var.name)
#                         ax.set_xlabel("Count")
#                     else:
#                         sns.barplot(
#                             x=label,
#                             y="count",
#                             data=df_plot,
#                             ax=ax,
#                             **kwargs,
#                         )
#                         # ax.set_xlabel(var.name)
#                         ax.set_ylabel("Count")

#                     ax.set_title(f"{var.name}  N={len(df_filtered)} ({filter_name})")
#                 # -----------------------
#                 # BACKWARD COMPATIBILITY (string)
#                 # -----------------------
#                 else:
#                     col = var
#                     #label = self.df_variables[col].values[0]

#                     if orient == "h":
#                         sns.countplot(y=col, data=df_filtered, ax=ax, **kwargs)
#                         #ax.set_ylabel(label)
#                         ax.set_xlabel("Count")
#                     else:
#                         sns.countplot(x=col, data=df_filtered, ax=ax, **kwargs)
#                         #ax.set_xlabel(label)
#                         ax.set_ylabel("Count")

#                     ax.set_title(f"{label}  N={len(df_filtered)} ({filter_name})")
                
#                 if j == 0:
#                     max_lines = wrap_yticks(ax, width=wrap_width)
#                     base_hspace = 0.25        # good for 1 line
#                     line_increment = 0.12    # extra space per additional line

#                     hspace = base_hspace + (max_lines - 1) * line_increment
                
#         fig.subplots_adjust(hspace=hspace)
#         if title:
#             fig.suptitle(title, fontsize=16, weight="bold")

#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         if path is not None:
#             plt.savefig(path)
#             plt.close()
#         else:
#             plt.show()

#     def estimate_row_height(
#         self, 
#         df,
#         var,
#         wrap_width=35,
#         base_per_category=0.35,
#         line_increment=0.12,
#         min_height=1.2,
#     ):
#         """
#         Estimate vertical inches needed for one row of countplots.
#         """
#         import textwrap

#         # number of categories actually plotted
#         n_categories = len(df.keys())

#         # wrap label and count lines
#         if isinstance(var, VariableSpec) and var.kind == "singlechoice":
#             if var.long_name in df.columns:
#                 longest_label = df[var.long_name].values[0]
#             else:
#                 longest_label = var.long_name

#         elif isinstance(var, VariableSpec) and var.kind == "multichoice":
#             longest_label = ""
#             for col in var.columns:
#                 option_label = self.df_variables[col].values[0].split("(")[0]
#                 if len(option_label) > len(longest_label):
#                     longest_label = option_label

#         wrapped = textwrap.fill(longest_label, width=wrap_width)
#         n_lines = wrapped.count("\n") + 1

#         height = (
#             n_categories * base_per_category
#             + (n_lines - 1) * line_increment
#         )

#         return max(height, min_height)
    
# def resolve_columns(variables):
#     """
#     Converts variables (str | VariableSpec) into a flat list of column names.
#     """
#     cols = []
#     for var in variables:
#         if isinstance(var, VariableSpec):
#             if var.kind == "singlechoice":
#                 cols.append(var.name)
#             elif var.kind == "singlechoice_text":
#                 cols.extend(var.columns)  # base + text columns
#             elif var.kind == "multichoice":
#                 cols.extend(var.columns)
#         else:
#             cols.append(var)
#     return cols
