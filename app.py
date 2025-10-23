import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import re
import itertools

# --- Stałe ---
ELECTRONEGATIVITY = {
    "H": 2.20, "Li": 0.98, "Na": 0.93, "Rb": 0.82,
    "Be": 1.57, "Mg": 1.31, "Ca": 1.00, "Sr": 0.95,
    "Sc": 1.36, "Y": 1.22,
    "La": 1.10, "Ce": 1.12, "Yb": 1.10, "Lu": 1.27,
    "Ac": 1.10, "Th": 1.30,
    "Zr": 1.33, "Hf": 1.30,
    "B": 2.04, "Al": 1.61, "Si": 1.90, "Ge": 2.01, "P": 2.19, "S": 2.58
}

ATOMIC_MASSES = {
    "H": 1.008, "Li": 6.94, "Na": 22.990, "Rb": 85.468,
    "Be": 9.0122, "Mg": 24.305, "Ca": 40.078, "Sr": 87.62,
    "Sc": 44.956, "Y": 88.906,
    "La": 138.91, "Ce": 140.12, "Yb": 173.05, "Lu": 174.97,
    "Ac": 227, "Th": 232.04,
    "Zr": 91.224, "Hf": 178.49,
    "B": 10.81, "Al": 26.982, "Si": 28.085, "Ge": 72.630, "P": 30.974, "S": 32.06
}

# Funkcje pomocnicze

def clean_formula(formula: str) -> str:
    """
    Usuwa '1' występujące bezpośrednio po symbolu pierwiastka, ale nie usuwa '1' będących częścią
    większej liczby (np. '10', '12').
    Przykłady:
      Li1Be1H8 -> LiBeH8
      C1H10 -> CH10
      LiBeH8 -> LiBeH8 (bez zmian)
    """
    return re.sub(r'([A-Z][a-z]?)(?:1)(?!\d)', r'\1', formula)

def normalize_formula(formula: str) -> str:
    """
    Normalizuje formułę do standardu: A < B < H
    Przykład: BeLuH8 -> LuBeH8, CaAlH12 -> AlCaH12
    """
    comp = parse_formula(formula)
    # wyciągnij A i B (wszystkie inne niż H)
    non_h = [(el, comp[el]) for el in comp if el != "H"]
    non_h_sorted = sorted(non_h, key=lambda x: x[0])  # sort alfabetycznie
    h_count = comp.get("H", 0)

    # złóż z powrotem
    normalized = ""
    for el, count in non_h_sorted:
        normalized += f"{el}{int(count) if count != 1 else ''}"
    if h_count:
        normalized += f"H{int(h_count) if h_count != 1 else ''}"
    return normalized

def parse_formula(formula):
    matches = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', formula)
    return {el: float(count) if count else 1.0 for el, count in matches}

def compute_en(comp):
    total_atoms = sum(comp.values())
    return sum(ELECTRONEGATIVITY.get(el, 0) * count for el, count in comp.items()) / total_atoms

def compute_hf(comp):
    n_H = comp.get("H", 0)
    n_total = sum(comp.values())
    return n_H / n_total if n_total else 0

def compute_mass_ratio(comp):
    m_H = comp.get("H", 0) * ATOMIC_MASSES.get("H", 1.008)
    m_X = sum(count * ATOMIC_MASSES.get(el, 0) for el, count in comp.items() if el != "H")
    return m_X / m_H if m_H else 0

def determine_region(en):
    # Nowy zakres S4
    if 2.00 <= en <= 2.10:
        return "S1"
    elif (1.90 <= en < 2.00) or (2.10 < en <= 2.20):
        return "S2"
    elif (1.80 <= en < 1.90) or (2.20 < en <= 2.30):
        return "S3"
    elif (1.70 <= en < 1.80) or (2.30 < en <= 2.40):
        return "S4"
    return "Outside"

def hf_to_color(hf):
    if hf >= 0.9: return "#004d00"
    elif hf >= 0.8: return "#1a661a"
    elif hf >= 0.7: return "#339933"
    elif hf >= 0.6: return "#66cc66"
    return "#b3ffb3"

def mass_to_marker(mass):
    if mass < 10: return 'o'
    elif mass < 20: return 's'
    elif mass < 30: return '^'
    return 'D'

# Gradient koloru regionów (alfa przezroczystości)
REGION_COLORS = {
    "S1": "rgba(255, 0, 0, 0.3)",           # transparentny czerwony
    "S2": "rgba(255, 165, 0, 0.3)",       # transparentny pomarańczowy
    "S3": "rgba(255, 230, 0, 0.3)",        # transparentny żółty
    "S4": "rgba(255, 255, 0, 0.25)"      # transparentny jasnożółty
}

# Mapowanie do matplotlib (konwersja rgba na tuple alfa)
def rgba_to_mpl_color(rgba_str):
    parts = rgba_str.replace("rgba(", "").replace(")", "").split(",")
    r, g, b, a = [float(p.strip()) for p in parts]
    return (r/255, g/255, b/255, a)

# --- Streamlit setup ---

st.set_page_config("Tc vs Electronegativity", layout="centered")
st.title("🧪 Assessing Chemical Composition for Ternary SuperHydrides")
st.markdown("Data-Driven Modeling of Superconducting Hydrides: From Composition to Critical Parameters")

# --- Wczytanie CSV ---
# Load CSV
try:
    df = pd.read_csv("Data-ternary.csv")
    # Dodajemy kolumnę z oczyszczonym formatem formuły (na wypadek)
    if 'formula' in df.columns:
        df['formula_clean'] = df['formula'].astype(str).apply(clean_formula)
    else:
        df['formula_clean'] = ""
except FileNotFoundError:
    st.error("❌ File `Data-ternary.csv` not found.")
    st.stop()

# Input
formula = st.text_input("Enter compound formula (e.g., AcAlH8):", "")

if formula:
    comp = parse_formula(formula)
    en = compute_en(comp)
    hf = compute_hf(comp)
    ratio = compute_mass_ratio(comp)
    region = determine_region(en)

    st.markdown(f"""
    ### ✅ Results for `{formula}`
    - **Electronegativity (avg)**: `{en:.3f}`
    - **Hydrogen Fraction (Hf)**: `{hf:.3f}`
    - **Mass Ratio (Mx/MH)**: `{ratio:.2f}`
    - **Region**: `{region}`
    """)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Region highlighting with new S4
ax.axvspan(2.00, 2.10, color=rgba_to_mpl_color(REGION_COLORS["S1"]), label="S1: High probability of Tc > 300 K")
ax.axvspan(1.90, 2.00, color=rgba_to_mpl_color(REGION_COLORS["S2"]), label="S2: High probability of Tc > 200 K")
ax.axvspan(2.10, 2.20, color=rgba_to_mpl_color(REGION_COLORS["S2"]))
ax.axvspan(1.80, 1.90, color=rgba_to_mpl_color(REGION_COLORS["S3"]), label="S3: Low probability of Tc > 200 K")
ax.axvspan(2.20, 2.30, color=rgba_to_mpl_color(REGION_COLORS["S3"]))
ax.axvspan(1.70, 1.80, color=rgba_to_mpl_color(REGION_COLORS["S4"]), label="S4: Very low probability of high-Tc")
ax.axvspan(2.30, 2.40, color=rgba_to_mpl_color(REGION_COLORS["S4"]))

# Data points
for _, row in df.iterrows():
    color = hf_to_color(row['hf'])
    marker = mass_to_marker(row['mass'])
    ax.scatter(row['en'], row['Tc'], color=color, marker=marker, edgecolor='black', s=70)




# Tworzymy adnotację, która początkowo jest niewidoczna
annot = ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

# Dane punktów do tooltipów
scatters = []
labels = []

for _, row in df.iterrows():
    color = hf_to_color(row['hf'])
    marker = mass_to_marker(row['mass'])
    sc = ax.scatter(row['en'], row['Tc'], color=color, marker=marker, edgecolor='black', s=70)
    scatters.append(sc)
    labels.append(row['formula_clean'] if 'formula_clean' in row else row.get('formula', 'Unknown'))

def update_annot(ind, sc):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = labels[ind["ind"][0]]
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor("yellow")
    annot.get_bbox_patch().set_alpha(0.8)

def hover(event):
    visible = annot.get_visible()
    if event.inaxes == ax:
        for sc in scatters:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind, sc)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
    if visible:
        annot.set_visible(False)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)




# New compound
if formula:
    ax.axvline(en, color='red', linestyle='--', linewidth=2, label=f"{formula} (new)")

# Axis and labels
ax.set_xlim(1.0, 3.25)
ax.set_ylim(0, 500)
ax.set_xlabel("Electronegativity")
ax.set_ylabel("Tc (K)")
ax.set_title("Tc vs Electronegativity with Hf and Mass Ratio")

# Legend – shapes and colors
hf_legend = [
    mpatches.Patch(color="#004d00", label="Hf ≥ 0.9"),
    mpatches.Patch(color="#1a661a", label="0.8 ≤ Hf < 0.9"),
    mpatches.Patch(color="#339933", label="0.7 ≤ Hf < 0.8"),
    mpatches.Patch(color="#66cc66", label="0.6 ≤ Hf < 0.7"),
    mpatches.Patch(color="#b3ffb3", label="Hf < 0.6"),
]

mass_legend = [
    mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Mx/MH < 10'),
    mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=10, label='10 ≤ Mx/MH < 20'),
    mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='20 ≤ Mx/MH < 30'),
    mlines.Line2D([], [], color='black', marker='D', linestyle='None', markersize=10, label='Mx/MH ≥ 30'),
]

region_legend = [
    mpatches.Patch(color=rgba_to_mpl_color(REGION_COLORS["S1"]), label='S1: High probability of Tc > 300 K'),
    mpatches.Patch(color=rgba_to_mpl_color(REGION_COLORS["S2"]), label='S2: High probability of Tc > 200 K'),
    mpatches.Patch(color=rgba_to_mpl_color(REGION_COLORS["S3"]), label='S3: Low probability of Tc > 200 K'),
    mpatches.Patch(color=rgba_to_mpl_color(REGION_COLORS["S4"]), label='S4: Very low probability of Tc > 200 K'),
]

line_legend = mlines.Line2D([], [], color='red', linestyle='--', label="Your compound")

# Dwie osobne legendy
left_legend = ax.legend(handles=region_legend, loc='upper left', fontsize=10, title="Regions")
ax.add_artist(left_legend)

# Pokazujemy legendę czerwonej linii tylko jeśli formuła wpisana
if formula:
    right_legend = ax.legend(
        handles=hf_legend + mass_legend + [line_legend],
        loc='upper right',
        fontsize=10,
        title="Legend"
    )
else:
    right_legend = ax.legend(
        handles=hf_legend + mass_legend,
        loc='upper right',
        fontsize=10,
        title="Legend"
    )

# Show plot
st.pyplot(fig)

# --- Sekcja: Szukanie w pełnej przestrzeni kombinacji ---

st.markdown("---")
st.header("🔍 Search Promising Ternary Hydrides in All Possible Element Combinations")

col1, col2 = st.columns(2)
with col1:
    hf_min = st.number_input("Hf_min", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    ratio_min = st.number_input("Mx/MH_min", min_value=0.0, value=0.0, step=0.1)
    en_min = st.number_input("Electronegativity_min", min_value=0.0, max_value=5.0, value=2.0, step=0.01)
with col2:
    hf_max = st.number_input("Hf_max", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    ratio_max = st.number_input("Mx/MH_max", min_value=0.0, value=25.0, step=0.1)
    en_max = st.number_input("Electronegativity_max", min_value=0.0, max_value=5.0, value=2.1, step=0.01)

elements = [el for el in ELECTRONEGATIVITY.keys() if el != "H" and ELECTRONEGATIVITY[el] is not None]
# --- Wybór zakresów stechiometrii ---
with st.expander("⚙️ Stoichiometry ranges (A_xB_yH_z)", expanded=False):
    x_raw = st.text_input("Values of x (comma-separated):", "1,2,3")
    y_raw = st.text_input("Values of y (comma-separated):", "1,2,3")
    z_raw = st.text_input("Values of z (comma-separated):", "2,4,6,8,9,10,12")


def parse_int_list(raw):
    """Konwertuje wpisany tekst (np. '1,2,3') na listę intów."""
    values = []
    for v in raw.split(","):
        v = v.strip()
        if v.isdigit():
            values.append(int(v))
    return values

x_values = parse_int_list(x_raw)
y_values = parse_int_list(y_raw)
z_values = parse_int_list(z_raw)

# Jeśli użytkownik zostawi pole puste, użyj domyślnych
if not x_values: x_values = [1, 2, 3]
if not y_values: y_values = [1, 2, 3]
if not z_values: z_values = [2, 4, 6, 8, 9, 10, 12]


compound_list = []

normalized_db = df['formula_clean'].apply(normalize_formula)
normalized_db_set = set(normalized_db.values)

for A, B in itertools.combinations(elements, 2):
    for x, y, z in itertools.product(x_values, y_values, z_values):
        chi = (x * ELECTRONEGATIVITY[A] + y * ELECTRONEGATIVITY[B] + z * ELECTRONEGATIVITY["H"]) / (x + y + z)
        hf_val = z / (x + y + z)
        mr_val = (x * ATOMIC_MASSES[A] + y * ATOMIC_MASSES[B]) / (z * ATOMIC_MASSES["H"])

        if hf_min <= hf_val <= hf_max and ratio_min <= mr_val <= ratio_max and en_min <= chi <= en_max:
            formula_raw = f"{A}{x}{B}{y}H{z}"
            formula_clean = clean_formula(formula_raw)
            in_csv = normalize_formula(formula_clean) in normalized_db_set
            compound_list.append((formula_clean, chi, hf_val, mr_val, "✅" if in_csv else "❌"))

# Stworzenie DataFrame
results_df = pd.DataFrame(compound_list, columns=["Formula", "Electronegativity", "Hf", "Mx/MH", "In Database"])

# Podsumowanie
st.markdown(f"### 📊 Found {len(results_df)} possible compounds matching criteria")

if not results_df.empty:
    st.dataframe(results_df)
    st.download_button("Download CSV", results_df.to_csv(index=False), "possible_compounds.csv", "text/csv")
else:
    st.warning("No compounds match the selected criteria.")

# --- Logowanie (opcjonalne) w konsoli dla kontroli (możesz wyłączyć później) ---
in_csv_count = results_df['In Database'].value_counts().get("✅", 0)
not_in_csv_count = results_df['In Database'].value_counts().get("❌", 0)
total_combinations = sum(1 for _ in itertools.product(elements, repeat=2)) * len(x_values) * len(y_values) * len(z_values)

print(f"Total combinations (approx): {total_combinations}")
print(f"Filtered combinations: {len(results_df)}")
print(f"Compounds in CSV: {in_csv_count}")
print(f"Compounds NOT in CSV: {not_in_csv_count}")

# --- Footer: Credits and Funding ---
st.markdown("---")

# Optional: Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ℹ️ Application Information")
    st.markdown(
        """
        This application is based on the publication:<br>
        <b>I. A. Wrona, P. Niegodajew, A. P. Durajski</b>, *High-temperature ternary superhydrides: A strategic roadmap to optimal superconducting parameters*,<br>
        <i>Adv. Funct. Mater.</i> 35, 2423680 (2025). **Please cite this work if you use this tool in your research.**<br><br>
        This research is funded by the <i>National Science Centre (Poland)</i><br>
        under Project No. 2022/47/B/ST3/00622.<br><br>
        Contact person:<br>
        <b>Artur Durajski</b><br>
        Częstochowa University of Technology<br>
        Email: <a href="mailto:artur.durajski@pcz.pl">artur.durajski@pcz.pl</a>
        """,
        unsafe_allow_html=True
    )


