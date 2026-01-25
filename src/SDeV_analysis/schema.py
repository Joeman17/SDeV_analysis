import re
from collections import defaultdict

class VariableSpec:
    def __init__(self, name, kind="singlechoice", columns=None, long_name=None):
        """
        name: logical base variable name
        kind: "singlechoice" or "multichoice"
        columns: list of column names (for multichoice)
        long_name: human-readable label (optional)
        """
        self.name = name
        self.kind = kind
        self.columns = columns or []
        self.long_name = long_name or name

    def __repr__(self):
        return f"VariableSpec(name={self.name!r}, kind={self.kind!r})"


class SurveySchema:
    def __init__(self, df_variables):
        """
        df_variables: DataFrame returned from the API 'variables' endpoint
        """
        self.df_variables = df_variables

        self.specs = self._init_specs()
        self._beautify_variable_labels()
        self._init_long_names()
        self.short_labels = {}
        self._build_short_labels()

    # ------------------------------------------------------------------
    # Spec initialization (extracted from Analysis.init_specs)
    # ------------------------------------------------------------------

    def _init_specs(self):
        """
        Initialize VariableSpec objects based on column structure.

        Rules:
        - only one column and it is Axyz_[0-9]+
            → text
        - Axyz exists and exactly one Axyz_[0-9]+ exists
            → singlechoice + free text
        - Axyz exists and >1 Axyz_[0-9]+ exist
            → multichoice
        - Axyz exists and no suffixed columns
            → singlechoice
        """


        suffix_pattern = re.compile(r"^(?P<base>.+)_(?P<idx>[0-9]+)$")

        all_cols = list(self.df_variables.keys())

        # --------------------------------------------------
        # group columns by base name
        # --------------------------------------------------
        groups = defaultdict(list)
        for col in all_cols:
            m = suffix_pattern.match(col)
            if m:
                groups[m.group("base")].append(col)
            else:
                groups[col]  # ensure base-only vars appear

        specs = {}

        for base, suffixed_cols in groups.items():
            has_base = base in all_cols
            n_suffix = len(suffixed_cols)
            # ----------------------------
            # Case 1: pure text
            # ----------------------------
            if not has_base and n_suffix == 1:
                specs[base] = VariableSpec(
                    name=base,
                    kind="text",
                    columns=suffixed_cols,
                )

            # ----------------------------
            # Case 2: singlechoice + text
            # ----------------------------
            elif has_base and n_suffix == 1:
                specs[base] = VariableSpec(
                    name=base,
                    kind="singlechoice_text",
                    columns=[base] + suffixed_cols,
                )

            # ----------------------------
            # Case 3: multichoice
            # ----------------------------
            elif has_base and n_suffix > 1:
                specs[base] = VariableSpec(
                    name=base,
                    kind="multichoice",
                    columns=sorted(suffixed_cols),
                )

            # ----------------------------
            # Case 4: normal singlechoice
            # ----------------------------
            elif has_base and n_suffix == 0:
                specs[base] = VariableSpec(
                    name=base,
                    kind="singlechoice",
                    columns=[base],
                )

            # ----------------------------
            # Anything else = malformed schema
            # ----------------------------
            else:
                raise ValueError(
                    f"Invalid variable structure for '{base}': "
                    f"base={has_base}, suffixes={suffixed_cols}"
                )

        return specs

    # ------------------------------------------------------------------
    # Label cleanup (extracted from Analysis.beautify_variable_labels)
    # ------------------------------------------------------------------

    def _beautify_variable_labels(self):
        """
        Mutates df_variables labels to split multichoice labels correctly.
        """
        for base, spec in self.specs.items():
            if spec.kind != "multichoice":
                continue

            # Base label: part before colon
            base_label = self.df_variables[base].label
            if isinstance(base_label, str) and ":" in base_label:
                self.df_variables[base].label = base_label.split(":", 1)[0]

            # Option labels: part after colon
            for col in spec.columns:
                col_label = self.df_variables[col].label
                if isinstance(col_label, str) and ":" in col_label:
                    self.df_variables[col].label = col_label.split(":", 1)[1]

    # ------------------------------------------------------------------
    # Long names (used heavily in plotting)
    # ------------------------------------------------------------------

    def _init_long_names(self):
        """
        Initialize human-readable long names for each VariableSpec.
        """
        for spec in self.specs.values():
            if spec.kind == "singlechoice":
                spec.long_name = self._safe_label(spec.name)
            else:
                spec.long_name = self._safe_label(spec.name)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _safe_label(self, col):
        """
        Return a cleaned label string for a variable.
        """
        try:
            label = self.df_variables[col].values[0]
            if isinstance(label, str):
                return label.split("(")[0].strip()
        except Exception:
            pass
        return col

    def get_spec(self, name):
        return self.specs.get(name)

    def is_multichoice(self, var):
        return isinstance(var, VariableSpec) and var.kind == "multichoice"

    def is_singlechoice(self, var):
        return isinstance(var, VariableSpec) and var.kind == "singlechoice"

    def resolve_columns(self, variables):
        """
        Resolve a list of variables (str or VariableSpec) into concrete columns.
        """
        cols = []

        for var in variables:
            if isinstance(var, VariableSpec):
                if var.kind == "multichoice":
                    cols.extend(var.columns)
                elif var.kind == "singlechoice_text":
                    cols.extend(var.columns)  # base + text columns
                else:
                    cols.append(var.name)
            else:
                cols.append(var)

        return cols
        
    def _build_short_labels(self):
        short = {}

        for var, row in self.df_variables.items():
            label = row.get("label")

            # manual overrides
            if var in MANUAL_SHORTNAMES:
                short[var] = MANUAL_SHORTNAMES[var]
                continue

            short[var] = make_short_label(label)

        self.short_labels = short

    def short_label(self, var):
        return self.short_labels.get(var, var)
    
def make_short_label(text, max_words=5):
    if not isinstance(text, str):
        return text

    # remove parentheses
    text = re.sub(r"\(.*?\)", "", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # replacements
    replacements = {
        "oder": "/",
        "und": "&",
        "beziehungsweise": "/",
        "kommunalpolitisch": "",
        "öffentliche oder gesellschaftliche": "öffentliche",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # cut after comma
    text = text.split(",")[0]

    # limit words
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    return text.strip()
    
MANUAL_SHORTNAMES = {
    "A502_01": "Frauen",
    "A502_02": "Migration",
}