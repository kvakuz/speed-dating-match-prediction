#| label: setup
#| include: false
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from IPython.display import Markdown, display

from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold, RandomizedSearchCV,
                                     cross_val_predict, cross_validate)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.inspection import partial_dependence
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                             recall_score, balanced_accuracy_score, confusion_matrix,
                             roc_curve, precision_recall_curve, make_scorer)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import statsmodels.api as sm
import shap
from scipy.stats import loguniform, randint, uniform

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DATA_PATH = "data/Speed Dating Data.csv"

# --- paleta i styl wykresów (paleta kategoryczna sprawdzona pod kątem daltonizmu)
C_F, C_M = "#eb6834", "#2a78d6"          # kobiety / mężczyźni
C_A, C_B = "#1baf7a", "#6250d6"          # scenariusz A / scenariusz B
C_GREY, C_INK, C_INK2 = "#898781", "#0b0b0b", "#52514e"
CAT6 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
CMAP_SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": C_INK2, "axes.titlecolor": C_INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": C_INK2, "ytick.color": C_INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
    "font.size": 10, "figure.dpi": 110, "lines.linewidth": 2,
    "axes.prop_cycle": plt.cycler(color=CAT6),
})

def pct(ax, axis="y", dec=0):
    fmt = mtick.PercentFormatter(1.0, decimals=dec)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)

def md_table(d, floatfmt=".3f", index=True):
    """Tabela Markdown z przecinkiem dziesiętnym."""
    d = d.copy()
    f = lambda v: (format(v, floatfmt).replace(".", ",") if isinstance(v, (float, np.floating)) and not pd.isna(v)
                   else ("" if isinstance(v, (float, np.floating)) else v))
    d = d.astype(object).map(f)
    if d.index.dtype == object:
        d.index = [f(v) for v in d.index]
    return Markdown(d.to_markdown(index=index))

def fnum(x, d=1):
    """Liczba w polskim formacie (przecinek dziesiętny)."""
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")

def fpct(x, d=1):
    return fnum(100 * x, d) + "%"

TRAITS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
TRAIT_PL = {"attr": "atrakcyjność", "sinc": "szczerość", "intel": "inteligencja",
            "fun": "poczucie humoru", "amb": "ambicja", "shar": "wspólne zainteresowania",
            "like": "ogólna sympatia", "prob": "szansa na „tak” partnera"}
INTERESTS = ["sports", "tvsports", "exercise", "dining", "museums", "art", "hiking",
             "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts",
             "music", "shopping", "yoga"]
RATINGS = TRAITS + ["like", "prob"]


# %%
#| label: load
raw = pd.read_csv(DATA_PATH, encoding="latin1")
n_rows, n_cols = raw.shape
n_people = raw["iid"].nunique()
n_waves = raw["wave"].nunique()
n_f = raw.loc[raw.gender == 0, "iid"].nunique()
n_m = raw.loc[raw.gender == 1, "iid"].nunique()
match_rate_raw = raw["match"].mean()


# %%
#| label: tbl-groups
#| tbl-cap: "Główne grupy zmiennych w zbiorze danych i ich wykorzystanie w projekcie"
groups = pd.DataFrame([
    ["Demografia", "`age`, `gender`, `race`, `field_cd`, `career_c`, `income`, `mn_sat`", "Ankieta przed wydarzeniem", "A, B"],
    ["Nastawienie", "`goal`, `date`, `go_out`, `imprace`, `imprelig`, `exphappy`", "Ankieta przed wydarzeniem", "A, B"],
    ["Zainteresowania", "17 zmiennych 1–10 (`sports`, `museums`, `clubbing`, …)", "Ankieta przed wydarzeniem", "A, B"],
    ["Deklarowane preferencje", "`attr1_1`…`shar1_1` (czego szukam), `attr2_1`… (czego szuka płeć przeciwna)", "Ankieta przed wydarzeniem", "A, B"],
    ["Samoocena", "`attr3_1`…`amb3_1` (1–10)", "Ankieta przed wydarzeniem", "A, B"],
    ["Cechy pary", "`int_corr`, `samerace`, `met`, różnica wieku (wyliczana)", "Przed wydarzeniem / karta randki", "A, B"],
    ["Oceny po randce", "`attr`…`shar`, `like`, `prob` oraz `*_o` (oceny wystawione przez partnera)", "Karta randki, w chwili decyzji", "B"],
    ["Decyzje", "`dec`, `dec_o`, `match`", "Karta randki", "**Y** / wyciek"],
    ["Ankiety po wydarzeniu", "zmienne z sufiksami `_s`, `_2`, `_3`, `match_es`, `you_call`, …", "W trakcie i po wydarzeniu", "wyciek, usunięte"],
], columns=["Grupa", "Zmienne", "Moment pomiaru", "Użycie"])
md_table(groups, index=False)


# %%
#| label: eda-prep
# lekka konwersja typów potrzebna do EDA (pełne czyszczenie w rozdz. 4)
eda = raw.copy()
for c in ["income", "tuition", "mn_sat"]:
    eda[c] = pd.to_numeric(eda[c].astype(str).str.replace(",", ""), errors="coerce")
people_raw = eda.groupby("iid").first()
people_raw["Płeć"] = people_raw["gender"].map({0: "Kobiety", 1: "Mężczyźni"})

# udziały preferencji (cecha / suma 6 cech × 100) – na poziomie osoby
def shares(d, suf):
    cols = [f"{t}{suf}" for t in TRAITS]
    return d[cols].div(d[cols].sum(axis=1, min_count=1), axis=0) * 100
pref1 = shares(people_raw, "1_1"); pref1.columns = TRAITS
pref2 = shares(people_raw, "2_1"); pref2.columns = TRAITS
w_rows = eda[eda.gender == 0]          # perspektywa kobiety = jedna randka raz


# %%
#| label: tbl-people
#| tbl-cap: "Charakterystyka uczestników (poziom osoby)"
def describe_people(g):
    return pd.Series({
        "Liczba osób": len(g),
        "Wiek: średnia": g["age"].mean(),
        "Wiek: odch. std.": g["age"].std(),
        "Wiek: min–max": f"{g['age'].min():.0f}–{g['age'].max():.0f}",
        "Rasa: European/Caucasian": (g["race"] == 2).mean(),
        "Rasa: Asian/Pacific Isl.": (g["race"] == 4).mean(),
        "Cel: „fun night out” lub „meet new people”": g["goal"].isin([1, 2]).mean(),
        "Cel: „get a date” lub „serious relationship”": g["goal"].isin([3, 4]).mean(),
        "Liczba randek (śr.)": eda[eda.iid.isin(g.index)].groupby("iid").size().mean(),
        "Odsetek „tak” (dec)": eda[eda.iid.isin(g.index)]["dec"].mean(),
    })
tab = pd.concat({k: describe_people(g) for k, g in people_raw.groupby("Płeć")}, axis=1).astype(object)
for r in ["Wiek: średnia", "Wiek: odch. std.", "Liczba randek (śr.)"]:
    tab.loc[r] = tab.loc[r].map(lambda v: fnum(v))
tab.loc["Liczba osób"] = tab.loc["Liczba osób"].map(lambda v: f"{int(v)}")
for r in ["Rasa: European/Caucasian", "Rasa: Asian/Pacific Isl.", "Cel: „fun night out” lub „meet new people”",
          "Cel: „get a date” lub „serious relationship”", "Odsetek „tak” (dec)"]:
    tab.loc[r] = tab.loc[r].map(lambda v: fpct(v))
md_table(tab)
dec_f = eda.loc[eda.gender == 0, "dec"].mean(); dec_m = eda.loc[eda.gender == 1, "dec"].mean()


# %%
#| label: fig-demo
#| fig-cap: "Rozkład wieku i deklarowanego celu udziału w wydarzeniu wg płci"
#| fig-width: 10
#| fig-height: 3.6
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
bins = np.arange(18, 56, 1)
for lab, col in [("Kobiety", C_F), ("Mężczyźni", C_M)]:
    ax.hist(people_raw.loc[people_raw["Płeć"] == lab, "age"].dropna(), bins=bins, alpha=0.55,
            color=col, label=lab, edgecolor="#fcfcfb", linewidth=0.8)
ax.set(title="Wiek uczestników", xlabel="wiek [lata]", ylabel="liczba osób"); ax.legend()
ax = axes[1]
goal_lab = {1: "fun night out", 2: "meet new people", 3: "get a date", 4: "serious relationship",
            5: "to say I did it", 6: "other"}
g = (people_raw.groupby("Płeć")["goal"].value_counts(normalize=True).unstack(0)
     .rename(index=goal_lab).reindex(list(goal_lab.values())))
y = np.arange(len(g)); h = 0.38
ax.barh(y - h/2, g["Kobiety"], h, color=C_F, label="Kobiety")
ax.barh(y + h/2, g["Mężczyźni"], h, color=C_M, label="Mężczyźni")
ax.set_yticks(y, g.index); ax.invert_yaxis(); pct(ax, "x")
ax.set(title="Główny cel udziału (goal)", xlabel="odsetek osób"); ax.legend(loc="lower right")
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()


# %%
#| label: fig-field
#| fig-cap: "Kierunki studiów uczestników (field_cd)"
#| fig-width: 9
#| fig-height: 4.6
field_lab = {1: "Prawo", 2: "Matematyka", 3: "Nauki społeczne/psychologia", 4: "Medycyna/farmacja/biotech",
             5: "Inżynieria", 6: "Filologia ang./dziennikarstwo", 7: "Historia/religia/filozofia",
             8: "Biznes/ekonomia/finanse", 9: "Edukacja", 10: "Biologia/chemia/fizyka",
             11: "Praca socjalna", 12: "Studia licencjackie/niezdecydowani", 13: "Politologia/stos. międzynar.",
             14: "Film", 15: "Sztuki piękne", 16: "Języki", 17: "Architektura", 18: "Inne"}
fc = people_raw.groupby("Płeć")["field_cd"].value_counts().unstack(0).fillna(0).rename(index=field_lab)
fc = fc.loc[fc.sum(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.barh(fc.index, fc["Kobiety"], color=C_F, label="Kobiety", edgecolor="#fcfcfb", linewidth=1)
ax.barh(fc.index, fc["Mężczyźni"], left=fc["Kobiety"], color=C_M, label="Mężczyźni", edgecolor="#fcfcfb", linewidth=1)
ax.set(xlabel="liczba osób", title="Kierunek studiów"); ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()


# %%
#| label: fig-balance
#| fig-cap: "Zbalansowanie klas: odsetek matchy ogółem, odsetek odpowiedzi „tak” wg płci i odsetek matchy w poszczególnych sesjach"
#| fig-width: 11
#| fig-height: 3.8
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [0.8, 0.8, 2.2]})
ax = axes[0]
vc = w_rows["match"].value_counts().sort_index()
bars = ax.bar(["brak matcha (0)", "match (1)"], vc.values, color=[C_GREY, C_B], width=0.6)
for b, v in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width()/2, v, f"{v}\n({v/vc.sum():.1%})".replace(".", ","), ha="center", va="bottom", fontsize=9)
ax.set(title="Klasy (poziom pary)", ylabel="liczba par", ylim=(0, vc.max() * 1.25)); ax.grid(axis="x", visible=False)
ax = axes[1]
dd = pd.Series({"Kobiety": dec_f, "Mężczyźni": dec_m})
bars = ax.bar(dd.index, dd.values, color=[C_F, C_M], width=0.6)
for b, v in zip(bars, dd.values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, fpct(v), ha="center", fontsize=9)
pct(ax); ax.set(title="Odsetek „tak” (dec)", ylim=(0, 0.6)); ax.grid(axis="x", visible=False)
ax = axes[2]
mw = w_rows.groupby("wave")["match"].agg(["mean", "size"])
ax.bar(mw.index.astype(str), mw["mean"], color=C_B, width=0.7)
ax.axhline(w_rows["match"].mean(), color=C_INK2, ls="--", lw=1)
ax.text(-0.6, w_rows["match"].mean() + 0.004, "średnia", ha="left", fontsize=8, color=C_INK2)
for i, (m, n) in enumerate(zip(mw["mean"], mw["size"])):
    ax.text(i, m + 0.005, str(n), ha="center", fontsize=7, color=C_INK2)
pct(ax); ax.set(title="Odsetek matchy wg sesji (nad słupkiem: liczba par)", xlabel="sesja (wave)")
ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.show()
n_pairs = len(w_rows); match_rate = w_rows["match"].mean()


# %%
#| label: fig-prefs
#| fig-cap: "Lewy panel: deklarowana ważność cech u partnera (udział w %, średnia ± 95% CI). Prawy panel: co płeć przeciwna faktycznie deklaruje, a co uczestnicy *myślą*, że deklaruje"
#| fig-width: 11
#| fig-height: 4
pref1["Płeć"] = people_raw["Płeć"]; pref2["Płeć"] = people_raw["Płeć"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
x = np.arange(6); h = 0.38
ax = axes[0]
for i, (lab, col) in enumerate([("Kobiety", C_F), ("Mężczyźni", C_M)]):
    d = pref1[pref1["Płeć"] == lab][TRAITS]
    m, se = d.mean(), d.sem()
    ax.bar(x + (i - 0.5) * h, m, h, yerr=1.96 * se, color=col, label=lab, capsize=2,
           error_kw={"elinewidth": 1, "ecolor": C_INK2})
ax.set_xticks(x, [TRAIT_PL[t] for t in TRAITS], rotation=25, ha="right")
ax.set(title="Czego szukam u partnera? (attr1_1 …)", ylabel="średni udział [% punktów]"); ax.legend()
ax.grid(axis="x", visible=False)
ax = axes[1]
# kobiety: co myślą, że cenią mężczyźni (pref2 kobiet) vs co cenią mężczyźni (pref1 mężczyzn)
pairs_ = [("Mężczyźni deklarują", pref1[pref1["Płeć"] == "Mężczyźni"][TRAITS].mean(), C_M, None),
          ("Kobiety sądzą, że mężczyźni cenią", pref2[pref2["Płeć"] == "Kobiety"][TRAITS].mean(), C_M, "//"),
          ("Kobiety deklarują", pref1[pref1["Płeć"] == "Kobiety"][TRAITS].mean(), C_F, None),
          ("Mężczyźni sądzą, że kobiety cenią", pref2[pref2["Płeć"] == "Mężczyźni"][TRAITS].mean(), C_F, "//")]
hw = 0.2
for i, (lab, m, col, hatch) in enumerate(pairs_):
    ax.bar(x + (i - 1.5) * hw, m, hw, color=col if hatch is None else "#fcfcfb", edgecolor=col,
           hatch=hatch, label=lab, linewidth=1)
ax.set_xticks(x, [TRAIT_PL[t] for t in TRAITS], rotation=25, ha="right")
ax.set(title="Deklaracje a wyobrażenia o płci przeciwnej (attr2_1 …)"); ax.legend(fontsize=8)
ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.show()
pf_mean = pref1.groupby("Płeć")[TRAITS].mean()


# %%
#| label: fig-ratings-match
#| fig-cap: "Odsetek matchy w zależności od oceny wystawionej po randce (linia pomarańczowa: ocena mężczyzny wystawiona przez kobietę; niebieska: ocena kobiety wystawiona przez mężczyznę). Pokazano poziomy ocen z co najmniej 20 parami"
#| fig-width: 11
#| fig-height: 5.6
fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharey=True)
for ax, r in zip(axes.ravel(), RATINGS):
    for col_, lab, c in [(r, "ocena wystawiona przez kobietę", C_F), (f"{r}_o", "ocena wystawiona przez mężczyznę", C_M)]:
        s = w_rows.assign(v=w_rows[col_].clip(1, 10).round()).groupby("v")["match"].agg(["mean", "size"])
        s = s[s["size"] >= 20]
        ax.plot(s.index, s["mean"], marker="o", ms=4, color=c, label=lab)
    ax.set_title(TRAIT_PL[r]); ax.set_xticks(range(1, 11, 2)); pct(ax)
axes[0, 0].set_ylabel("odsetek matchy"); axes[1, 0].set_ylabel("odsetek matchy")
for ax in axes[1]: ax.set_xlabel("ocena (1–10)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
plt.tight_layout(); plt.show()


# %%
#| label: fig-corr
#| fig-cap: "Macierz korelacji Pearsona: oceny wystawione przez kobietę (_K), przez mężczyznę (_M) i match"
#| fig-width: 9
#| fig-height: 7.4
cc = w_rows[RATINGS + [f"{r}_o" for r in RATINGS] + ["int_corr", "match"]].copy()
cc.columns = [f"{r}_K" for r in RATINGS] + [f"{r}_M" for r in RATINGS] + ["int_corr", "match"]
corr = cc.corr()
fig, ax = plt.subplots(figsize=(9, 7.4))
sns.heatmap(corr, cmap=CMAP_DIV, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 6.5},
            linewidths=0.5, linecolor="#fcfcfb", cbar_kws={"shrink": 0.7, "label": "r Pearsona"}, ax=ax)
ax.set_title("Korelacje ocen po randce"); ax.grid(False)
plt.tight_layout(); plt.show()


# %%
#| label: fig-pre-match
#| fig-cap: "Odsetek matchy w zależności od wybranych cech pary znanych przed randką (słupki błędów: 95% CI; liczby: liczba par)"
#| fig-width: 11
#| fig-height: 3.8
def rate_ci(s):
    p, n = s.mean(), s.size
    return p, 1.96 * np.sqrt(p * (1 - p) / n), n
tmp = w_rows.assign(age_diff=w_rows["age_o"] - w_rows["age"])
fig, axes = plt.subplots(1, 4, figsize=(11, 3.8), sharey=True, gridspec_kw={"width_ratios": [1.6, 1.6, 0.7, 0.7]})
specs = [
    (pd.cut(tmp["age_diff"], [-40, -4, -2, 0, 2, 4, 6, 40], labels=["≤−4", "−3…−2", "−1…0", "1…2", "3…4", "5…6", "≥7"]),
     "Różnica wieku (M − K) [lata]"),
    (pd.cut(tmp["int_corr"], [-1, -0.2, 0, 0.2, 0.4, 0.6, 1], labels=["<−0,2", "−0,2…0", "0…0,2", "0,2…0,4", "0,4…0,6", ">0,6"]),
     "Korelacja zainteresowań (int_corr)"),
    (tmp["samerace"].map({0: "nie", 1: "tak"}), "Ta sama rasa"),
    (np.where(tmp["met"] == 1, "tak", np.where(tmp["met_o"] == 1, "tak", "nie")), "Znali się wcześniej"),
]
for ax, (grp, title) in zip(axes, specs):
    st = tmp.groupby(grp, observed=True)["match"].apply(rate_ci).apply(pd.Series)
    st.columns = ["p", "ci", "n"]
    ax.bar(st.index.astype(str), st["p"], yerr=st["ci"], color=C_A, width=0.65, capsize=2,
           error_kw={"elinewidth": 1, "ecolor": C_INK2})
    for i, (p, n) in enumerate(zip(st["p"], st["n"])):
        ax.text(i, 0.005, f"{int(n)}", ha="center", fontsize=7, color="#ffffff")
    ax.axhline(match_rate, color=C_INK2, ls="--", lw=1)
    ax.set_title(title, fontsize=10); ax.tick_params(axis="x", rotation=35); ax.grid(axis="x", visible=False)
pct(axes[0]); axes[0].set_ylabel("odsetek matchy")
plt.tight_layout(); plt.show()


# %%
#| label: fig-missing
#| fig-cap: "Odsetek braków danych w wybranych zmiennych (poziom osoby dla ankiety przed wydarzeniem, poziom randki dla ocen)"
#| fig-width: 10
#| fig-height: 4.4
miss_p = people_raw[["mn_sat", "tuition", "income", "expnum", "attr5_1", "attr4_1", "career_c", "shar1_1",
                     "attr3_1", "age", "exphappy", "goal", "race", "sports"]].isna().mean()
miss_r = w_rows[["shar", "shar_o", "amb", "amb_o", "fun", "prob", "intel", "sinc", "like", "attr", "int_corr"]].isna().mean()
miss_l = raw[["attr1_3", "you_call", "attr7_2", "satis_2", "attr1_s"]].isna().mean()
fig, axes = plt.subplots(1, 3, figsize=(10, 4.4), sharex=True, gridspec_kw={"width_ratios": [1.3, 1, 0.6]})
for ax, s, title, col in [(axes[0], miss_p, "Ankieta przed wydarzeniem (osoby)", C_A),
                          (axes[1], miss_r, "Karta randki (pary)", C_B),
                          (axes[2], miss_l, "Ankiety późniejsze\n(wyciek, usuwane)", C_GREY)]:
    s = s.sort_values()
    ax.barh(s.index, s.values, color=col, height=0.65)
    for i, v in enumerate(s.values):
        ax.text(v + 0.01, i, fpct(v, 0), va="center", fontsize=8)
    ax.set_title(title, fontsize=10); pct(ax, "x"); ax.set_xlim(0, 1.05); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()


# %%
#| label: fig-outliers
#| fig-cap: "Rozkłady wybranych zmiennych liczbowych (wykresy pudełkowe; punkty: obserwacje poza 1,5 × IQR)"
#| fig-width: 11
#| fig-height: 3.4
box = {"wiek": people_raw["age"], "różnica wieku M−K": tmp["age_diff"], "int_corr": w_rows["int_corr"],
       "ocena attr (K→M)": w_rows["attr"], "ocena like (M→K)": w_rows["like_o"],
       "samoocena attr3_1": people_raw["attr3_1"], "oczekiwane szczęście": people_raw["exphappy"]}
fig, axes = plt.subplots(1, len(box), figsize=(11, 3.4))
out_rows = []
for ax, (k, s) in zip(axes, box.items()):
    s = s.dropna()
    ax.boxplot(s, widths=0.55, patch_artist=True, boxprops=dict(facecolor="#cde2fb", edgecolor=C_M),
               medianprops=dict(color=C_INK), whiskerprops=dict(color=C_M), capprops=dict(color=C_M),
               flierprops=dict(marker="o", ms=3, mfc=C_F, mec="none", alpha=0.6))
    ax.set_title(k, fontsize=9); ax.set_xticks([]); ax.grid(axis="x", visible=False)
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    out_rows.append([k, s.min(), s.max(), ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum(), len(s)])
plt.tight_layout(); plt.show()
out_tab = pd.DataFrame(out_rows, columns=["Zmienna", "Min", "Max", "Liczba obs. poza 1,5×IQR", "N"])
n_zero_r = int((raw[RATINGS] == 0).sum().sum())


# %%
#| label: leakage
leak = (["dec", "dec_o", "match_es"]
        + [c for c in raw.columns if re.search(r"(_s|_2|_3)$", c)]
        + ["you_call", "them_cal", "length", "numdat_2"]
        + ["id", "idg", "round", "position", "positin1", "order", "partner"]
        + ["field", "undergra", "from", "zipcode", "career"])
leak = list(dict.fromkeys([c for c in leak if c in raw.columns]))
df = raw.drop(columns=leak).copy()


# %%
#| label: clean
# 1) liczby zapisane jako tekst z przecinkami
for c in ["income", "tuition", "mn_sat"]:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
# 2) met: 1 = znali się, 0/2 = nie, pozostałe wartości = błąd -> NaN
for c in ["met", "met_o"]:
    df[c] = np.where(df[c] == 1, 1, np.where(df[c].isin([0, 2]), 0, np.nan))
# 3) oceny po randce: wartości spoza skali 1-10 przycinamy
for c in RATINGS + [r + "_o" for r in RATINGS]:
    df[c] = df[c].clip(lower=1, upper=10)

# 4) tabela osób – cechy sprzed wydarzenia (stałe dla danej osoby)
person_cols = (["gender", "wave", "age", "race", "field_cd", "career_c", "goal", "date", "go_out",
                "imprace", "imprelig", "exphappy", "income", "mn_sat"] + INTERESTS
               + [f"{t}1_1" for t in TRAITS] + [f"{t}2_1" for t in TRAITS]
               + [f"{t}3_1" for t in ["attr", "sinc", "intel", "fun", "amb"]])
people = df.groupby("iid")[person_cols].first()
people[INTERESTS] = people[INTERESTS].clip(lower=1)

# 5) preferencje -> udziały procentowe (różne skale w sesjach 6-9, sumy != 100)
for suf in ["1_1", "2_1"]:
    cols = [f"{t}{suf}" for t in TRAITS]
    people[cols] = people[cols].div(people[cols].sum(axis=1, min_count=1), axis=0) * 100

# 6) odwrócenie skal porządkowych: wyższa wartość = częściej
people["date_freq"] = 8 - people["date"]
people["go_out_freq"] = 8 - people["go_out"]
people = people.drop(columns=["date", "go_out"])

# 7) flagi braków dla zmiennych o nielosowych brakach
people["income_missing"] = people["income"].isna().astype(int)
people["mn_sat_missing"] = people["mn_sat"].isna().astype(int)

# 8) łączenie rzadkich kategorii (< 10 osób)
def lump(s, min_n=10, other=99):
    vc = s.value_counts()
    return s.where(~s.isin(vc[vc < min_n].index), other)
people["field_cd"] = lump(people["field_cd"])
people["career_c"] = lump(people["career_c"])
people["race"] = lump(people["race"], other=6)       # Native American (brak/nieliczni) -> Other
pref_sum_before = raw.groupby("iid")[[f"{t}1_1" for t in TRAITS]].first().sum(axis=1)


# %%
#| label: pairs
wdf = df[df.gender == 0].reset_index(drop=True)
base = wdf[["iid", "pid", "wave", "match", "int_corr", "samerace", "condtn"]].copy()
base["met_any"] = ((wdf["met"] == 1) | (wdf["met_o"] == 1)).astype(int)
pf = people.drop(columns=["gender", "wave"]).add_suffix("_f")
pm = people.drop(columns=["gender", "wave"]).add_suffix("_m")
ratings = pd.concat({**{f"{r}_by_f": wdf[r] for r in RATINGS},
                     **{f"{r}_by_m": wdf[f"{r}_o"] for r in RATINGS}}, axis=1)
pair = pd.concat([base, ratings], axis=1).join(pf, on="iid").join(pm, on="pid")

# kontrola spójności
mdf = df[df.gender == 1]
keys_f = set(zip(wdf.iid, wdf.pid)); keys_m = set(zip(mdf.pid, mdf.iid))
check = pd.DataFrame({
    "Wartość": [len(wdf), int(wdf.pid.isna().sum()), len(keys_f & keys_m), int(mdf.pid.isna().sum()),
                int((~wdf.pid.isin(people.index)).sum()), int(pair.duplicated(["iid", "pid"]).sum())]},
    index=["Liczba par (wiersze kobiet)", "Brakujące pid po stronie kobiet",
           "Pary obecne w obu perspektywach", "Wiersze mężczyzn bez pid (sesja 5)",
           "Pary, których pid nie ma w tabeli osób", "Zduplikowane pary"])


# %%
#| label: tbl-check
#| tbl-cap: "Kontrola poprawności budowy zbioru par"
md_table(check, floatfmt=".0f")


# %%
#| label: features
T5 = ["attr", "sinc", "intel", "fun", "amb"]
new = pd.DataFrame(index=pair.index)
new["age_diff"] = pair["age_m"] - pair["age_f"]
new["age_absdiff"] = new["age_diff"].abs()
new["interest_absdiff"] = np.abs(pair[[f"{i}_f" for i in INTERESTS]].values
                                 - pair[[f"{i}_m" for i in INTERESTS]].values).mean(axis=1)
new["same_field"] = (pair["field_cd_f"] == pair["field_cd_m"]).astype(int)
new["same_goal"] = (pair["goal_f"] == pair["goal_m"]).astype(int)
def pref_fit(pref_side, self_side):
    P = pair[[f"{t}1_1_{pref_side}" for t in T5]].values
    S = pair[[f"{t}3_1_{self_side}" for t in T5]].values
    return (P * S).sum(axis=1) / P.sum(axis=1)
new["pref_fit_f"] = pref_fit("f", "m")
new["pref_fit_m"] = pref_fit("m", "f")
new["samerace_x_imprace_f"] = pair["samerace"] * pair["imprace_f"]
new["samerace_x_imprace_m"] = pair["samerace"] * pair["imprace_m"]
def wrating(side):
    P = pair[[f"{t}1_1_{side}" for t in TRAITS]].values
    R = pair[[f"{t}_by_{side}" for t in TRAITS]].values
    ok = ~np.isnan(R)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(P * np.where(ok, R, 0), axis=1) / np.where(ok, P, 0).sum(axis=1)
new["wrating_by_f"] = wrating("f")
new["wrating_by_m"] = wrating("m")
pair = pd.concat([pair, new], axis=1)

CAT = [f"{c}_{s}" for c in ["race", "field_cd", "career_c", "goal"] for s in "fm"]
ID = ["iid", "pid", "wave", "match"]
B_ONLY = [f"{r}_by_{s}" for r in RATINGS for s in "fm"] + ["wrating_by_f", "wrating_by_m"]
FEAT_A = [c for c in pair.columns if c not in ID + B_ONLY]
FEAT_B = FEAT_A + B_ONLY
NUM_A = [c for c in FEAT_A if c not in CAT]
NUM_B = [c for c in FEAT_B if c not in CAT]
pair[CAT] = pair[CAT].astype("float")


# %%
#| label: tbl-final
#| tbl-cap: "Ostateczne zbiory cech w obu scenariuszach"
summ = pd.DataFrame({
    "Scenariusz A (przed randką)": [len(FEAT_A), len(NUM_A), len(CAT)],
    "Scenariusz B (A + oceny po randce)": [len(FEAT_B), len(NUM_B), len(CAT)]},
    index=["Liczba zmiennych", "w tym liczbowe/binarne", "w tym nominalne (one-hot)"])
md_table(summ, floatfmt=".0f")


# %%
#| label: fig-univariate
#| fig-cap: "Jednowymiarowa siła związku cech z matchem: 20 cech scenariusza A i 10 cech scenariusza B o największej |r| (korelacja punktowo-dwuseryjna)"
#| fig-width: 11
#| fig-height: 5.2
def side(c):
    if c.endswith("_f") or "_by_f" in c: return "kobieta"
    if c.endswith("_m") or "_by_m" in c: return "mężczyzna"
    return "para"
side_col = {"kobieta": C_F, "mężczyzna": C_M, "para": C_A}
r_all = pair[NUM_B].corrwith(pair["match"]).dropna()
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1]})
for ax, cols, k, title in [(axes[0], NUM_A, 20, "Scenariusz A (przed randką)"),
                           (axes[1], B_ONLY, 10, "Scenariusz B: oceny po randce")]:
    r = r_all.reindex(cols).dropna()
    r = r.reindex(r.abs().sort_values(ascending=False).index[:k])[::-1]
    ax.barh(r.index, r.values, color=[side_col[side(c)] for c in r.index], height=0.65)
    ax.axvline(0, color=C_INK2, lw=0.8); ax.set_title(title); ax.set_xlabel("r z match")
    ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=v, label=k) for k, v in side_col.items()], loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.03), title=None)
plt.tight_layout(); plt.show()
rA_max = r_all.reindex(NUM_A).abs().max(); rB_max = r_all.reindex(B_ONLY).abs().max()


# %%
#| label: split
outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
tr_idx, te_idx = next(outer.split(pair, pair["match"], groups=pair["wave"]))
train = pair.iloc[tr_idx].reset_index(drop=True)
test = pair.iloc[te_idx].reset_index(drop=True)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
assert set(train.iid) & set(test.iid) == set() and set(train.pid) & set(test.pid) == set()

rows = [["Zbiór uczący", len(train), train.wave.nunique(), train.iid.nunique() + train.pid.nunique(), train.match.mean(),
         ", ".join(map(str, sorted(train.wave.unique())))],
        ["Zbiór testowy", len(test), test.wave.nunique(), test.iid.nunique() + test.pid.nunique(), test.match.mean(),
         ", ".join(map(str, sorted(test.wave.unique())))]]
for k, (a, b) in enumerate(cv.split(train, train.match, groups=train.wave), 1):
    v = train.iloc[b]
    rows.append([f"CV – fold walidacyjny {k}", len(v), v.wave.nunique(), v.iid.nunique() + v.pid.nunique(),
                 v.match.mean(), ", ".join(map(str, sorted(v.wave.unique())))])
split_tab = pd.DataFrame(rows, columns=["Zbiór", "Pary", "Sesje", "Osoby", "Odsetek matchy", "Sesje (wave)"])


# %%
#| label: tbl-split
#| tbl-cap: "Podział na zbiór uczący i testowy oraz foldy walidacji krzyżowej (wszystkie grupowane po sesjach)"
md_table(split_tab, index=False)


# %%
#| label: model-defs
def make_prep(num, cat, scale=True):
    num_steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)],
        verbose_feature_names_out=False)

spw = (1 - train.match.mean()) / train.match.mean()
MODELS = {
    "Regresja logistyczna": (LogisticRegression(max_iter=5000, solver="liblinear", random_state=RANDOM_STATE), True,
        {"clf__C": loguniform(1e-3, 10), "clf__penalty": ["l1", "l2"], "clf__class_weight": [None, "balanced"]}, 30),
    "Drzewo decyzyjne": (DecisionTreeClassifier(random_state=RANDOM_STATE), False,
        {"clf__max_depth": randint(2, 12), "clf__min_samples_leaf": randint(5, 150), "clf__criterion": ["gini", "entropy"],
         "clf__class_weight": [None, "balanced"], "clf__ccp_alpha": loguniform(1e-5, 1e-2)}, 40),
    "SVM liniowy": (LinearSVC(dual="auto", max_iter=20000, random_state=RANDOM_STATE), True,
        {"clf__C": loguniform(1e-4, 1), "clf__class_weight": [None, "balanced"]}, 20),
    "SVM RBF": (SVC(kernel="rbf", random_state=RANDOM_STATE), True,
        {"clf__C": loguniform(1e-1, 1e2), "clf__gamma": loguniform(1e-4, 3e-2), "clf__class_weight": [None, "balanced"]}, 20),
    "Las losowy": (RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE), False,
        {"clf__max_depth": [None, 6, 10, 16], "clf__min_samples_leaf": randint(1, 30),
         "clf__max_features": ["sqrt", 0.2, 0.4], "clf__class_weight": [None, "balanced_subsample"]}, 15),
    "XGBoost": (XGBClassifier(n_estimators=400, tree_method="hist", eval_metric="logloss", n_jobs=2,
                              random_state=RANDOM_STATE, verbosity=0), False,
        {"clf__learning_rate": loguniform(0.01, 0.2), "clf__max_depth": randint(2, 7), "clf__min_child_weight": randint(1, 20),
         "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.3, 0.7),
         "clf__reg_lambda": loguniform(0.1, 20), "clf__scale_pos_weight": [1, spw]}, 30),
}
SCEN = {"A": (NUM_A, CAT), "B": (NUM_B, CAT)}

def get_scores(est, X):
    """Wynik ciągły modelu: prawdopodobieństwo (gdy dostępne) lub odległość od hiperpłaszczyzny (SVM)."""
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)


# %%
#| label: tuning
search, fit_time = {}, {}
for scen, (num, cat) in SCEN.items():
    X, y, g = train[num + cat], train["match"], train["wave"]
    for name, (clf, scale, grid, n_iter) in MODELS.items():
        t0 = time.time()
        pipe = Pipeline([("prep", make_prep(num, cat, scale)), ("clf", clf)])
        rs = RandomizedSearchCV(pipe, grid, n_iter=n_iter, cv=cv, random_state=RANDOM_STATE,
                                scoring={"ap": "average_precision", "auc": "roc_auc"}, refit="ap",
                                n_jobs=1 if name in ["Las losowy", "XGBoost"] else -1)
        rs.fit(X, y, groups=g)
        search[(scen, name)] = rs
        fit_time[(scen, name)] = time.time() - t0


# %%
#| label: tbl-cv
#| tbl-cap: "Wyniki walidacji krzyżowej (5 foldów grupowanych po sesjach) dla najlepszych konfiguracji hiperparametrów: średnia ± odchylenie standardowe między foldami"
def fmt_params(p):
    out = []
    for k, v in p.items():
        k = k.replace("clf__", "")
        if isinstance(v, float): v = f"{v:.3g}"
        out.append(f"{k}={v}")
    return ", ".join(out)
cv_rows = []
for (scen, name), rs in search.items():
    i = rs.best_index_; r = rs.cv_results_
    cv_rows.append({"Scenariusz": scen, "Model": name,
                    "CV PR AUC": f"{fnum(r['mean_test_ap'][i], 3)} ± {fnum(r['std_test_ap'][i], 3)}",
                    "CV ROC AUC": f"{fnum(r['mean_test_auc'][i], 3)} ± {fnum(r['std_test_auc'][i], 3)}",
                    "Liczba konfiguracji": len(r["params"]),
                    "Najlepsze hiperparametry": fmt_params(rs.best_params_)})
cv_tab = pd.DataFrame(cv_rows)
md_table(cv_tab, index=False)


# %%
#| label: oof
def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = np.nanargmax(f1[:-1])
    return t[i], f1[i]

final = {}
for (scen, name), rs in search.items():
    num, cat = SCEN[scen]
    Xtr, Xte = train[num + cat], test[num + cat]
    est = rs.best_estimator_
    method = "predict_proba" if hasattr(est, "predict_proba") else "decision_function"
    oof = cross_val_predict(clone(est), Xtr, train["match"], groups=train["wave"], cv=cv, method=method, n_jobs=1)
    oof = oof[:, 1] if oof.ndim == 2 else oof
    thr, f1_oof = best_threshold(train["match"], oof)
    s = get_scores(est, Xte)
    pred = (s >= thr).astype(int)
    final[(scen, name)] = dict(est=est, oof=oof, thr=thr, s=s, pred=pred, f1_oof=f1_oof)


# %%
#| label: test-metrics
rng = np.random.default_rng(RANDOM_STATE)
boot_idx = [rng.integers(0, len(test), len(test)) for _ in range(1000)]
yte = test["match"].values
def boot_ci(metric, s):
    vals = [metric(yte[b], s[b]) for b in boot_idx if yte[b].min() != yte[b].max()]
    return np.percentile(vals, [2.5, 97.5])

res_rows = []
for (scen, name), d in final.items():
    s, pred = d["s"], d["pred"]
    auc, ap = roc_auc_score(yte, s), average_precision_score(yte, s)
    auc_ci, ap_ci = boot_ci(roc_auc_score, s), boot_ci(average_precision_score, s)
    res_rows.append({"Scenariusz": scen, "Model": name, "ROC AUC": auc, "ROC_lo": auc_ci[0], "ROC_hi": auc_ci[1],
                     "PR AUC": ap, "PR_lo": ap_ci[0], "PR_hi": ap_ci[1],
                     "F1": f1_score(yte, pred), "Precyzja": precision_score(yte, pred, zero_division=0),
                     "Czułość": recall_score(yte, pred), "Zbal. trafność": balanced_accuracy_score(yte, pred)})
res = pd.DataFrame(res_rows)
best_cv = {sc: max([k for k in search if k[0] == sc], key=lambda k: search[k].best_score_)[1] for sc in "AB"}


# %%
#| label: tbl-test
#| tbl-cap: "Wyniki na zbiorze testowym (niewidziane sesje). Przedziały ufności: bootstrap 1000 prób. Metryki progowe przy progu maksymalizującym F1 na predykcjach out-of-fold. Odniesienie: klasyfikator losowy ma ROC AUC = 0,5 i PR AUC ≈ odsetek matchy w teście"
res_show = res.copy()
res_show.insert(3, "ROC AUC 95% CI", [f"{fnum(a, 3)}–{fnum(b, 3)}" for a, b in zip(res.ROC_lo, res.ROC_hi)])
res_show.insert(5, "PR AUC 95% CI", [f"{fnum(a, 3)}–{fnum(b, 3)}" for a, b in zip(res.PR_lo, res.PR_hi)])
md_table(res_show.drop(columns=["ROC_lo", "ROC_hi", "PR_lo", "PR_hi"]), index=False)


# %%
#| label: fig-test-bars
#| fig-cap: "Porównanie modeli na zbiorze testowym: ROC AUC i PR AUC (słupki błędów: 95% bootstrapowy przedział ufności), scenariusz A (przed randką) i B (z ocenami po randce)"
#| fig-width: 11
#| fig-height: 3.9
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
names = list(MODELS); x = np.arange(len(names)); h = 0.38
for ax, met, ref in [(axes[0], "ROC AUC", 0.5), (axes[1], "PR AUC", test.match.mean())]:
    pre = met.split()[0]
    for i, (sc, col) in enumerate([("A", C_A), ("B", C_B)]):
        d = res[res.Scenariusz == sc].set_index("Model").loc[names]
        lo = d[met] - d[f"{pre}_lo"]
        hi = d[f"{pre}_hi"] - d[met]
        ax.bar(x + (i - 0.5) * h, d[met], h, yerr=[lo, hi], color=col, label=f"Scenariusz {sc}", capsize=2,
               error_kw={"elinewidth": 1, "ecolor": C_INK2})
        for xi, v in zip(x + (i - 0.5) * h, d[met]):
            ax.text(xi, 0.02, f"{v:.2f}".replace(".", ","), ha="center", fontsize=7.5, color="#ffffff", rotation=90, va="bottom")
    ax.axhline(ref, color=C_INK2, ls="--", lw=1, label="klasyfikator losowy")
    ax.set_xticks(x, names, rotation=20, ha="right"); ax.set_title(met, pad=8); ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
plt.tight_layout(); plt.show()


# %%
#| label: fig-roc-pr
#| fig-cap: "Krzywe ROC i precyzja–czułość na zbiorze testowym dla obu scenariuszy"
#| fig-width: 11
#| fig-height: 8.2
fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
for j, sc in enumerate("AB"):
    for name, col in zip(names, CAT6):
        s = final[(sc, name)]["s"]
        fpr, tpr, _ = roc_curve(yte, s); p, r, _ = precision_recall_curve(yte, s)
        lw = 2.4 if name == best_cv[sc] else 1.4
        axes[0, j].plot(fpr, tpr, color=col, lw=lw, label=f"{name} ({roc_auc_score(yte, s):.3f})".replace(".", ","))
        axes[1, j].plot(r, p, color=col, lw=lw, label=f"{name} ({average_precision_score(yte, s):.3f})".replace(".", ","))
    axes[0, j].plot([0, 1], [0, 1], color=C_GREY, ls="--", lw=1)
    axes[1, j].axhline(yte.mean(), color=C_GREY, ls="--", lw=1)
    axes[0, j].set(title=f"Krzywa ROC – scenariusz {sc}", xlabel="odsetek fałszywie pozytywnych (FPR)", ylabel="czułość (TPR)")
    axes[1, j].set(title=f"Krzywa precyzja–czułość – scenariusz {sc}", xlabel="czułość", ylabel="precyzja", ylim=(0, 1.02))
    axes[0, j].legend(loc="lower right", title="ROC AUC", fontsize=8); axes[1, j].legend(loc="upper right", title="PR AUC", fontsize=8)
plt.tight_layout(); plt.show()


# %%
#| label: fig-cm
#| fig-cap: "Macierze pomyłek na zbiorze testowym dla modelu wybranego w walidacji krzyżowej (najwyższa CV PR AUC) w każdym scenariuszu"
#| fig-width: 9
#| fig-height: 3.8
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, sc in zip(axes, "AB"):
    d = final[(sc, best_cv[sc])]
    cm = confusion_matrix(yte, d["pred"])
    sns.heatmap(cm, annot=False, cmap=CMAP_SEQ, cbar=False, ax=ax, linewidths=2, linecolor="#fcfcfb")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j + 0.5, i + 0.5, f"{v}\n({v / cm[i].sum():.0%})", ha="center", va="center",
                color="#ffffff" if v > cm.max() * 0.4 else C_INK, fontsize=10)
    ax.set_xticklabels(["brak matcha", "match"]); ax.set_yticklabels(["brak matcha", "match"], rotation=0)
    ax.set(xlabel="predykcja", ylabel="rzeczywistość", title=f"Scenariusz {sc}: {best_cv[sc]}\n(próg = {fnum(d['thr'], 3)})")
plt.tight_layout(); plt.show()


# %%
#| label: split-exp
schemes = {
    "losowy": (StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), None),
    "grupy: kobieta (iid)": (StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE), "iid"),
    "grupy: sesja (wave)": (cv, "wave"),
}
sp_rows = []
for sc in "AB":
    num, cat = SCEN[sc]
    for name in ["Regresja logistyczna", "XGBoost"]:
        est = search[(sc, name)].best_estimator_
        for sch, (splitter, gcol) in schemes.items():
            out = cross_validate(clone(est), train[num + cat], train["match"],
                                 groups=None if gcol is None else train[gcol], cv=splitter,
                                 scoring={"auc": "roc_auc", "ap": "average_precision"}, n_jobs=1)
            sp_rows.append({"Scenariusz": sc, "Model": name, "Podział": sch,
                            "ROC AUC": out["test_auc"].mean(), "PR AUC": out["test_ap"].mean()})
sp = pd.DataFrame(sp_rows)


# %%
#| label: fig-split-exp
#| fig-cap: "Średnia CV ROC AUC przy trzech schematach podziału danych. Podział losowy zawyża wyniki, bo te same osoby trafiają do części uczącej i walidacyjnej"
#| fig-width: 10
#| fig-height: 3.6
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
sch_col = {"losowy": C_F, "grupy: kobieta (iid)": "#eda100", "grupy: sesja (wave)": C_B}
for ax, sc in zip(axes, "AB"):
    d = sp[sp.Scenariusz == sc]
    x = np.arange(2); hh = 0.26
    for i, sch in enumerate(schemes):
        v = d[d["Podział"] == sch].set_index("Model").loc[["Regresja logistyczna", "XGBoost"], "ROC AUC"]
        ax.bar(x + (i - 1) * hh, v, hh, color=sch_col[sch], label=sch)
        for xi, vv in zip(x + (i - 1) * hh, v):
            ax.text(xi, vv + 0.01, f"{vv:.3f}".replace(".", ","), ha="center", fontsize=8)
    ax.set_xticks(x, ["Regresja logistyczna", "XGBoost"]); ax.set_title(f"Scenariusz {sc}")
    ax.set_ylim(0.45, 1); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("CV ROC AUC"); axes[0].legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()
sp_p = sp.pivot_table(index=["Scenariusz", "Model"], columns="Podział", values="ROC AUC")


# %%
#| label: imbalance
num, cat = SCEN["B"]
Xtr = train[num + cat]
lr_p = {k.replace("clf__", ""): v for k, v in search[("B", "Regresja logistyczna")].best_params_.items() if k != "clf__class_weight"}
xgb_p = {k.replace("clf__", ""): v for k, v in search[("B", "XGBoost")].best_params_.items() if k != "clf__scale_pos_weight"}
base_models = {
    "Regresja logistyczna": (lambda w: LogisticRegression(max_iter=5000, solver="liblinear", random_state=RANDOM_STATE,
                                                          class_weight="balanced" if w else None, **lr_p), True),
    "XGBoost": (lambda w: XGBClassifier(n_estimators=400, tree_method="hist", eval_metric="logloss", n_jobs=2,
                                        random_state=RANDOM_STATE, verbosity=0, scale_pos_weight=spw if w else 1, **xgb_p), False),
}
strategies = {"brak korekty": (None, False), "wagi klas": (None, True),
              "SMOTE": (SMOTE(random_state=RANDOM_STATE), False),
              "undersampling": (RandomUnderSampler(random_state=RANDOM_STATE), False)}
scoring = {"auc": "roc_auc", "ap": "average_precision", "f1": "f1",
           "prec": make_scorer(precision_score, zero_division=0), "rec": "recall", "bacc": "balanced_accuracy"}
imb_rows = []
for mname, (factory, scale) in base_models.items():
    for sname, (sampler, w) in strategies.items():
        steps = [("prep", make_prep(num, cat, scale))] + ([("sampler", sampler)] if sampler is not None else []) + [("clf", factory(w))]
        out = cross_validate(ImbPipeline(steps), Xtr, train["match"], groups=train["wave"], cv=cv, scoring=scoring, n_jobs=1)
        imb_rows.append({"Model": mname, "Strategia": sname, **{k: out[f"test_{k}"].mean() for k in scoring}})
imb = pd.DataFrame(imb_rows).rename(columns={"auc": "ROC AUC", "ap": "PR AUC", "f1": "F1 (0,5)", "prec": "Precyzja (0,5)",
                                             "rec": "Czułość (0,5)", "bacc": "Zbal. trafność (0,5)"})


# %%
#| label: tbl-imb
#| tbl-cap: "Strategie radzenia sobie z niezbalansowaniem: średnie wyniki 5-krotnej CV grupowanej po sesjach (scenariusz B). Metryki progowe przy progu 0,5"
md_table(imb, index=False)


# %%
#| label: fig-imb
#| fig-cap: "Wpływ strategii niezbalansowania na metryki rankingowe (ROC AUC, PR AUC) i progowe przy progu 0,5 (czułość, precyzja, F1); CV grupowana po sesjach, scenariusz B"
#| fig-width: 11
#| fig-height: 3.8
mets = ["ROC AUC", "PR AUC", "Czułość (0,5)", "Precyzja (0,5)", "F1 (0,5)"]
strat_col = dict(zip(strategies, [C_GREY, C_M, C_F, C_A]))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, mname in zip(axes, base_models):
    d = imb[imb.Model == mname].set_index("Strategia")
    x = np.arange(len(mets)); hh = 0.2
    for i, sname in enumerate(strategies):
        ax.bar(x + (i - 1.5) * hh, d.loc[sname, mets], hh, color=strat_col[sname], label=sname)
    ax.set_xticks(x, mets, rotation=15); ax.set_title(mname); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
plt.tight_layout(); plt.show()


# %%
#| label: shap-calc
def feature_source(name, cols):
    """Mapuje nazwę cechy po one-hot (np. 'race_f_2.0') na zmienną źródłową ('race_f')."""
    if name in cols: return name
    for c in CAT:
        if name.startswith(c + "_"): return c
    return name

def group_of(src):
    if src in B_ONLY:
        return "Oceny wystawione przez kobietę" if "_by_f" in src else "Oceny wystawione przez mężczyznę"
    if src.endswith("_f"): return "Cechy kobiety (sprzed randki)"
    if src.endswith("_m"): return "Cechy mężczyzny (sprzed randki)"
    return "Cechy pary"

shap_res = {}
for sc in "AB":
    est = search[(sc, "XGBoost")].best_estimator_
    num, cat = SCEN[sc]
    Xt = est.named_steps["prep"].transform(test[num + cat])
    fn = est.named_steps["prep"].get_feature_names_out()
    expl = shap.TreeExplainer(est.named_steps["clf"])
    sv = expl(pd.DataFrame(Xt, columns=fn))
    shap_res[sc] = dict(sv=sv, fn=fn, Xt=Xt, est=est)

def polish_shap():
    """Polskie opisy paska kolorów na wykresie beeswarm."""
    for a in plt.gcf().axes:
        if a.get_ylabel() == "Feature value":
            a.set_ylabel("wartość cechy"); a.set_yticks(a.get_yticks(), ["niska", "wysoka"])
        else:
            labs = [l.get_text() for l in a.get_yticklabels()]
            if any(l.startswith("Sum of") for l in labs):
                labs = [re.sub(r"Sum of (\d+) other features", r"pozostałe cechy (\1)", l) for l in labs]
                a.set_yticks(a.get_yticks(), labs)

def nice(n):
    """Czytelna etykieta cechy na wykresach."""
    m = re.match(r"^(attr|sinc|intel|fun|amb|shar|like|prob)_by_(f|m)$", n)
    if m: return f"{m.group(1)}: ocena {'K→M' if m.group(2) == 'f' else 'M→K'}"
    m = re.match(r"^wrating_by_(f|m)$", n)
    if m: return f"ocena ważona deklaracjami {'K→M' if m.group(1) == 'f' else 'M→K'}"
    m = re.match(r"^(.+)_(f|m)_(\d+)\.0$", n)          # zmienna po one-hot, np. race_f_2.0
    if m: return f"{m.group(1)} (K) = {m.group(3)}" if m.group(2) == "f" else f"{m.group(1)} (M) = {m.group(3)}"
    m = re.match(r"^(.+)_(f|m)$", n)
    if m: return f"{m.group(1)} ({'K' if m.group(2) == 'f' else 'M'})"
    return n


# %%
#| label: fig-shap-b
#| fig-cap: "Wartości SHAP dla modelu XGBoost w scenariuszu B (zbiór testowy): 20 najważniejszych cech. Każdy punkt to jedna para; kolor oznacza wartość cechy (czerwony: wysoka, niebieski: niska); oś pozioma pokazuje wpływ na logit prawdopodobieństwa matcha"
#| fig-width: 10
#| fig-height: 7
svB = shap_res["B"]["sv"]
svB_n = shap.Explanation(svB.values, svB.base_values, data=svB.data, feature_names=[nice(f) for f in shap_res["B"]["fn"]])
plt.figure(figsize=(10, 7))
shap.plots.beeswarm(svB_n, max_display=20, show=False, color=CMAP_DIV, plot_size=None); polish_shap()
plt.title("SHAP – scenariusz B (XGBoost)", fontsize=11, fontweight="bold"); plt.xlabel("wartość SHAP (wpływ na logit P(match))")
plt.tight_layout(); plt.show()


# %%
#| label: fig-shap-groups
#| fig-cap: "Suma średnich |SHAP| w grupach zmiennych (scenariusz B, zbiór testowy): czyja perspektywa i jaki typ informacji decyduje o matchu"
#| fig-width: 9
#| fig-height: 3.4
mabs = pd.Series(np.abs(svB.values).mean(axis=0), index=shap_res["B"]["fn"])
src = mabs.groupby(lambda n: feature_source(n, FEAT_B)).sum()
grp = src.groupby(group_of).sum().sort_values()
gcol = {"Oceny wystawione przez kobietę": C_F, "Oceny wystawione przez mężczyznę": C_M,
        "Cechy kobiety (sprzed randki)": "#f5b597", "Cechy mężczyzny (sprzed randki)": "#9ec5f4", "Cechy pary": C_A}
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.barh(grp.index, grp.values, color=[gcol[g] for g in grp.index], height=0.6)
for i, v in enumerate(grp.values):
    ax.text(v + 0.01, i, f"{v:.2f} ({v / grp.sum():.0%})".replace(".", ","), va="center", fontsize=9)
ax.set(xlabel="suma średnich |SHAP|", title="Udział grup zmiennych w predykcjach"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, grp.max() * 1.3)
plt.tight_layout(); plt.show()


# %%
#| label: fig-shap-a
#| fig-cap: "Wartości SHAP dla modelu XGBoost w scenariuszu A (tylko dane sprzed randki): 15 najważniejszych cech"
#| fig-width: 10
#| fig-height: 5.6
svA = shap_res["A"]["sv"]
svA_n = shap.Explanation(svA.values, svA.base_values, data=svA.data, feature_names=[nice(f) for f in shap_res["A"]["fn"]])
plt.figure(figsize=(10, 5.6))
shap.plots.beeswarm(svA_n, max_display=15, show=False, color=CMAP_DIV, plot_size=None); polish_shap()
plt.title("SHAP – scenariusz A (XGBoost)", fontsize=11, fontweight="bold"); plt.xlabel("wartość SHAP (wpływ na logit P(match))")
plt.tight_layout(); plt.show()


# %%
#| label: dec-model
dec_df = raw[["iid", "gender", "dec"] + TRAITS].dropna().copy()
for c in TRAITS:
    dec_df[c] = dec_df[c].clip(1, 10)
coef_rows, dec_models = [], {}
for gval, glab in [(0, "Kobiety"), (1, "Mężczyźni")]:
    d = dec_df[dec_df.gender == gval]
    Xs = (d[TRAITS] - d[TRAITS].mean()) / d[TRAITS].std()
    m = sm.Logit(d["dec"], sm.add_constant(Xs)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["iid"]})
    dec_models[glab] = m
    ci = m.conf_int()
    for t in TRAITS:
        coef_rows.append({"Płeć": glab, "Cecha": TRAIT_PL[t], "t": t, "Współczynnik": m.params[t],
                          "CI_lo": ci.loc[t, 0], "CI_hi": ci.loc[t, 1], "p": m.pvalues[t], "OR": np.exp(m.params[t])})
coef = pd.DataFrame(coef_rows)

# test różnic między płciami: model łączny z interakcjami gender × ocena
dz = dec_df.copy()
for c in TRAITS:
    dz[c] = dz.groupby("gender")[c].transform(lambda s: (s - s.mean()) / s.std())
    dz[f"{c}_x_M"] = dz[c] * dz["gender"]
Xi = sm.add_constant(dz[TRAITS + ["gender"] + [f"{c}_x_M" for c in TRAITS]])
mi = sm.Logit(dz["dec"], Xi).fit(disp=0, cov_type="cluster", cov_kwds={"groups": dz["iid"]})
inter = pd.DataFrame({"Różnica (M − K)": mi.params[[f"{c}_x_M" for c in TRAITS]].values,
                      "p-wartość": mi.pvalues[[f"{c}_x_M" for c in TRAITS]].values},
                     index=[TRAIT_PL[c] for c in TRAITS])


# %%
#| label: fig-dec-coef
#| fig-cap: "Regresja logistyczna decyzji indywidualnej (dec) na standaryzowanych ocenach partnera: współczynniki z 95% CI (odpornymi na skupienia po osobie)"
#| fig-width: 9
#| fig-height: 3.8
fig, ax = plt.subplots(figsize=(9, 3.8))
y0 = np.arange(len(TRAITS))
for k, (glab, col) in enumerate([("Kobiety", C_F), ("Mężczyźni", C_M)]):
    d = coef[coef["Płeć"] == glab].set_index("t").loc[TRAITS]
    yy = y0 + (k - 0.5) * 0.3
    ax.errorbar(d["Współczynnik"], yy, xerr=[d["Współczynnik"] - d["CI_lo"], d["CI_hi"] - d["Współczynnik"]],
                fmt="o", color=col, ms=7, capsize=3, label=glab, elinewidth=1.5)
ax.axvline(0, color=C_INK2, lw=0.8)
ax.set_yticks(y0, [TRAIT_PL[t] for t in TRAITS]); ax.invert_yaxis()
ax.set(xlabel="współczynnik logitowy na 1 odch. std. oceny", title="Co zwiększa szansę na „tak”?"); ax.legend()
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()


# %%
#| label: tbl-inter
#| tbl-cap: "Różnice między płciami we wpływie ocen na decyzję: współczynniki interakcji płeć × ocena w modelu łącznym (dodatni = cecha ważniejsza dla mężczyzn)"
md_table(inter)


# %%
#| label: fig-declared
#| fig-cap: "Deklarowana a ujawniona ważność sześciu cech partnera (udziały sumujące się do 100% w obrębie każdej miary i płci)"
#| fig-width: 11
#| fig-height: 4
fnB = list(shap_res["B"]["fn"]); mabsB = np.abs(svB.values).mean(axis=0)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
comp = {}
for ax, (glab, sfx) in zip(axes, [("Kobiety", "f"), ("Mężczyźni", "m")]):
    declared = pf_mean.loc[glab, TRAITS] / 100
    cdec = coef[coef["Płeć"] == glab].set_index("t").loc[TRAITS, "Współczynnik"].abs()
    revealed = cdec / cdec.sum()
    sh = pd.Series({t: mabsB[fnB.index(f"{t}_by_{sfx}")] for t in TRAITS}); sh = sh / sh.sum()
    comp[glab] = pd.DataFrame({"deklarowana": declared, "decyzje (dec)": revealed, "model matcha (SHAP)": sh})
    x = np.arange(6); hh = 0.27
    for i, (lab, col) in enumerate([("deklarowana", C_GREY), ("decyzje (dec)", C_F if sfx == "f" else C_M),
                                    ("model matcha (SHAP)", C_B)]):
        ax.bar(x + (i - 1) * hh, comp[glab][lab], hh, color=col, label=lab)
    ax.set_xticks(x, [TRAIT_PL[t] for t in TRAITS], rotation=25, ha="right"); pct(ax)
    ax.set_title(f"{glab}: ważność cech partnera"); ax.grid(axis="x", visible=False); ax.legend(fontsize=8)
axes[0].set_ylabel("udział w ważności")
plt.tight_layout(); plt.show()


# %%
#| label: fig-moderation
#| fig-cap: "Czy osoby deklarujące wysoką wagę danej cechy rzeczywiście bardziej się nią kierują? Odsetek „tak” w zależności od oceny partnera, osobno dla osób z dolnej i górnej trójki (tercyla) deklarowanej wagi tej cechy"
#| fig-width: 11
#| fig-height: 3.6
pref_person = pref1[TRAITS]
dd_ = raw[["iid", "gender", "dec"] + TRAITS].copy()
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
mod_rows = []
for ax, t in zip(axes, ["attr", "intel", "sinc"]):
    rk = pref_person[t].groupby(people_raw["gender"]).rank(pct=True)
    q = pd.Series(np.where(rk <= 1/3, "dolny", np.where(rk > 2/3, "górny", "środkowy")), index=rk.index)
    q[rk.isna()] = np.nan
    dd_["tercyl"] = dd_["iid"].map(q)
    dd_["v"] = dd_[t].clip(1, 10).round()
    for terc, ls in [("dolny", "--"), ("górny", "-")]:
        s = dd_[dd_["tercyl"] == terc].groupby("v")["dec"].agg(["mean", "size"]); s = s[s["size"] >= 30]
        ax.plot(s.index, s["mean"], ls=ls, marker="o", ms=4, color=C_B if terc == "górny" else C_GREY,
                label=f"{terc} tercyl deklarowanej wagi")
    ax.set(title=f"{TRAIT_PL[t]}", xlabel="ocena partnera (1–10)"); pct(ax)
axes[0].set_ylabel("odsetek „tak”"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()


# %%
#| label: fig-pdp
#| fig-cap: "Wykresy częściowej zależności dla modelu XGBoost w scenariuszu B (zbiór uczący): przewidywane prawdopodobieństwo matcha w funkcji wybranych cech"
#| fig-width: 11
#| fig-height: 5.6
estB = search[("B", "XGBoost")].best_estimator_
XtrB = train[NUM_B + CAT]
pdp_feats = ["attr_by_f", "attr_by_m", "like_by_f", "like_by_m", "shar_by_f", "prob_by_m", "int_corr", "age_diff"]
fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharey=True)
pdp_max = 0
for ax, f in zip(axes.ravel(), pdp_feats):
    cgrid = {"int_corr": np.linspace(-0.6, 0.9, 31), "age_diff": np.arange(-8, 13)}.get(f, np.arange(1, 11))
    pdres = partial_dependence(estB, XtrB, [f], kind="average", grid_resolution=30)
    grid_, avg = pdres["grid_values"][0], pdres["average"][0]
    col = C_F if f.endswith("_f") else (C_M if f.endswith("_m") else C_A)
    ax.plot(grid_, avg, color=col, lw=2.2)
    v = XtrB[f].dropna(); v = v[(v >= cgrid.min()) & (v <= cgrid.max())]
    ax.plot(v.sample(min(300, len(v)), random_state=1), np.full(min(300, len(v)), 0.005), "|", color=C_GREY, alpha=0.4, ms=8)
    ax.set_title(nice(f), fontsize=10); pct(ax); pdp_max = max(pdp_max, avg.max())
axes[0, 0].set_ylim(0, pdp_max * 1.1)
axes[0, 0].set_ylabel("średnie P(match)"); axes[1, 0].set_ylabel("średnie P(match)")
plt.tight_layout(); plt.show()


# %%
#| label: fig-pdp2d
#| fig-cap: "Dwuwymiarowy wykres częściowej zależności: ogólna sympatia wyrażona przez kobietę i przez mężczyznę. Match wymaga wysokiej oceny od obu stron"
#| fig-width: 6.4
#| fig-height: 5
pd2 = partial_dependence(estB, XtrB, ["like_by_f", "like_by_m"], kind="average",
                         custom_values={"like_by_f": np.arange(1, 11), "like_by_m": np.arange(1, 11)})
gx, gy = pd2["grid_values"]
fig, ax = plt.subplots(figsize=(6.4, 5))
im = ax.imshow(pd2["average"][0].T, origin="lower", cmap=CMAP_SEQ, aspect="auto",
               extent=[gx.min() - 0.5, gx.max() + 0.5, gy.min() - 0.5, gy.max() + 0.5])
cb = plt.colorbar(im, ax=ax); cb.set_label("średnie P(match)"); cb.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set(xlabel="like: ocena wystawiona przez kobietę", ylabel="like: ocena wystawiona przez mężczyznę",
       title="PDP 2D: like K→M × like M→K"); ax.grid(False)
plt.tight_layout(); plt.show()
pd2_hi = pd2["average"][0].max(); pd2_lo = pd2["average"][0].min()


# %%
#| label: cp-select
dB = final[("B", "XGBoost")]
pB, thrB = estB.predict_proba(test[NUM_B + CAT])[:, 1], dB["thr"]
tp = np.where((yte == 1) & (pB >= thrB))[0]; fn_ = np.where((yte == 1) & (pB < thrB))[0]; tn = np.where((yte == 0) & (pB < thrB))[0]
sel = {"para A: match wykryty (TP)": tp[np.argsort(pB[tp])[len(tp) // 2]],
       "para B: match przeoczony (FN)": fn_[np.argmin(pB[fn_])],
       "para C: brak matcha (TN)": tn[np.argsort(pB[tn])[len(tn) // 2]]}
show_cols = ["attr_by_f", "like_by_f", "fun_by_f", "shar_by_f", "attr_by_m", "like_by_m", "fun_by_m", "shar_by_m", "prob_by_m", "int_corr", "age_diff"]
sel_tab = test.loc[list(sel.values()), show_cols].copy()
sel_tab.insert(0, "P(match)", pB[list(sel.values())]); sel_tab.insert(0, "match", yte[list(sel.values())])
sel_tab.index = list(sel.keys())


# %%
#| label: tbl-cp
#| tbl-cap: "Wybrane pary ze zbioru testowego: rzeczywisty wynik, przewidywane prawdopodobieństwo i kluczowe oceny z randki"
md_table(sel_tab.T, floatfmt=".2f")


# %%
#| label: fig-cp
#| fig-cap: "Profile ceteris paribus dla trzech par ze zbioru testowego (XGBoost, scenariusz B). Kropka to faktyczna wartość cechy i aktualna predykcja; linia przerywana to próg decyzyjny"
#| fig-width: 11
#| fig-height: 3.6
cp_feats = ["attr_by_f", "like_by_f", "attr_by_m", "like_by_m"]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
pair_col = dict(zip(sel, [C_B, C_F, C_GREY]))
for ax, f in zip(axes, cp_feats):
    grid_ = np.arange(1, 11)
    for lab, i in sel.items():
        row = test.loc[[i], NUM_B + CAT]
        rep = pd.concat([row] * len(grid_), ignore_index=True); rep[f] = grid_
        pr = estB.predict_proba(rep)[:, 1]
        ax.plot(grid_, pr, color=pair_col[lab], label=lab, lw=2)
        if not np.isnan(row[f].iloc[0]):
            ax.plot(row[f].iloc[0], pB[i], "o", color=pair_col[lab], ms=8, mec="#fcfcfb", mew=2)
    ax.axhline(thrB, color=C_INK2, ls="--", lw=1)
    ax.set(title=nice(f), xlabel="wartość oceny"); ax.set_xticks(range(1, 11, 2)); pct(ax)
axes[0].set_ylabel("P(match)")
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
plt.tight_layout(); plt.show()


# %%
#| label: fig-waterfall
#| fig-cap: "Dekompozycja SHAP (waterfall) predykcji dla pary A (match wykryty) i pary B (match przeoczony). Wartości w skali logitu; E[f(x)] to średnia predykcja modelu"
#| fig-width: 11
#| fig-height: 5
for lab in ["para A: match wykryty (TP)", "para B: match przeoczony (FN)"]:
    i = sel[lab]
    e = shap.Explanation(svB.values[i], svB.base_values[i], data=np.round(svB.data[i], 2), feature_names=svB_n.feature_names)
    plt.figure(figsize=(9, 4.6))
    shap.plots.waterfall(e, max_display=12, show=False)
    plt.title(lab, fontsize=11, fontweight="bold"); plt.tight_layout(); plt.show()
