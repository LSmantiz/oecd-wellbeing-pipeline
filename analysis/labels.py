"""Human-readable labels and concept grouping for OECD indicator codes.

Ported from the pretty_names mapping in the original R script. Kept as a module
rather than inlined in the document so that every plot uses the same labels.
"""

import textwrap

PRETTY_NAMES: dict[str, str] = {
    "EMP_RA": "Employment rate",
    "SR_ELD_RA_T": "Sex ratio 65+ (male/female)",
    "INCOME_DISP": "Disposable income",
    "UNEM_RA": "Unemployment rate",
    "SUBJ_SOC_SUPP": "Subjective social support",
    "SUBJ_PERC_CORR": "Subjective perceived corruption",
    "POP_DEN_GR_T": "Population density growth index (2001 vs. 2014/2015)",
    "ROOMS_PC": "Rooms per capita",
    "AIR_POL": "Air pollution (PM2.5)",
    "DEATH_RA_M": "Death rate (male)",
    "POP_TOT_GI_T": "Population growth index (2001 vs. 2014/2015)",
    "KID_WOM_RA_T": "Child-to-woman ratio",
    "BB_ACC": "Household broadband access",
    "YOU_DEP_RA_T": "Youth dependency rate",
    "SR_TOT_RA_T": "Sex ratio total population (male/female)",
    "EDU38_SH": "Labour force with at least secondary education (%)",
    "SUBJ_LIFE_SAT": "Subjective well-being",
    # selected by the Python pipeline but not by the original R run
    "HOMIC_RA": "Homicide rate",
    "VOTERS_SH": "Voter turnout (%)",
}

# Indicators from the OECD regional demography database. The original emphasises
# these in bold in the importance plot, since the paper argues that demographic
# structure matters more than the literature suggests.
DEMOGRAPHIC: set[str] = {
    "SR_ELD_RA_T",
    "SR_TOT_RA_T",
    "DEATH_RA_M",
    "POP_TOT_GI_T",
    "POP_DEN_GR_T",
    "KID_WOM_RA_T",
    "YOU_DEP_RA_T",
}

# Indicator families, for tracking stability at the level of the underlying
# concept rather than the specific measure.
#
# This matters because the richer datasets on the completeness frontier contain
# finer-grained variants: EMP_RA is displaced by measures such as EMP_SH_PT_F
# (female part-time employment share). At the indicator level that reads as
# "employment stopped being selected", when in fact a more specific measure of
# employment won. Grouping separates the two readings.
#
# Longest prefixes are matched first, so UNEM is not swallowed by EMP and
# SUBJ_LIFE_SAT is excluded before the SUBJ family is considered.
CONCEPT: dict[str, str] = {
    "UNEM": "Unemployment",
    "INCOME": "Income",
    "DEATH": "Mortality",
    "HOMIC": "Crime",
    "VOTERS": "Civic participation",
    "ROOMS": "Housing",
    "EMP": "Employment",
    "EDU": "Education",
    "POP": "Population structure",
    "SR": "Sex ratio",
    "YOU": "Age structure",
    "KID": "Age structure",
    "AIR": "Air quality",
    "SUBJ": "Subjective measures",
    "BB": "Broadband access",
}

TARGET = "SUBJ_LIFE_SAT"


def label(code: str, width: int | None = None) -> str:
    """Readable label for an indicator code, optionally wrapped."""
    name = PRETTY_NAMES.get(code, code)
    return "\n".join(textwrap.wrap(name, width)) if width else name


def labels(codes, width: int | None = None) -> list[str]:
    return [label(c, width) for c in codes]


def is_demographic(code: str) -> bool:
    return code in DEMOGRAPHIC


def concept(code: str) -> str:
    """Group an indicator code into its underlying concept.

    Prefixes are tried longest-first so that more specific families win: without
    that, UNEM_RA would need EMP to be checked after UNEM, and the result would
    depend on dict ordering rather than on the rule.

    Codes matching no family fall back to their readable label, so an ungrouped
    indicator is still identifiable rather than silently bucketed.
    """
    if code == TARGET:
        return "Subjective well-being (target)"

    for prefix in sorted(CONCEPT, key=len, reverse=True):
        if code.startswith(prefix):
            return CONCEPT[prefix]

    return PRETTY_NAMES.get(code, code)