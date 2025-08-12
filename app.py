import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import re
import itertools

# --- Stałe ---
ELECTRONEGATIVITY = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04,
    "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Ti": 1.54,
    "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91,
    "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55,
    "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60,
    "Mo": 2.16, "Ru": 2.20, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78,
    "Sn": 1.96, "Sb": 2.05, "I": 2.66, "Cs": 0.79, "Ba": 0.89, "La": 1.10,
    "Hf": 1.30, "Ac": 1.10
}

ATOMIC_MASSES = {
    "H": 1.008, "Li": 6.94, "Be": 9.01, "B": 10.81, "C": 12.01, "N": 14.01,
    "O": 16.00, "F": 18.998, "Na": 22.99, "Mg": 24.31, "Al": 26.98, "Si": 28.09,
    "P": 30.97, "S": 32.07, "Cl": 35.45, "K": 39.10, "Ca": 40.08, "Ti": 47.87,
    "V": 50.94, "Cr": 52.00, "Mn": 54.94, "Fe": 55.85, "Co": 58.93, "Ni": 58.69,
    "Cu": 63.55, "Zn": 65.38, "Ga": 69.72, "Ge": 72.63, "As": 74.92, "Se": 78.96,
    "Br": 79.90, "Rb": 85.47, "Sr": 87.62, "Y": 88.91, "Zr": 91.22, "Nb": 92.91,
    "Mo": 95.95, "Ru": 101.07, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82,
    "Sn": 118.71, "Sb": 121.76, "I": 126.90, "Cs": 132.91, "Ba": 137.33, "La": 138.91,
    "Hf": 178.49, "Ac": 227.03
}

# --- Funkcje pomocnicze ---

def parse_formula(formula):
    """Parsuje wzór chemiczny do słownika element:ilość."""
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
    if 2.00 <= en <= 2.10:
        return "S1"
    elif (1.90 <= en < 2.00) or (2.10 < en <= 2.20):
        return "S2"
    elif (1.80 <= en < 1.90) or (2.20 < en <= 2.30):
        return "S3"
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

def clean_formula(formula):
    """Usuwa jedynki z nazwy związku np. Li1Be1H8 -> LiBeH8.
    Nie usuwa liczb większych niż 1, np. H10 zostaje."""
    def repl(match):
        el = match.group(1)
        num = match.group(2)
        if num == '1' or num == '':
            return el
        return f"{el}{num}"
    return re.sub(r'([A-Z][a-z]*)(\d*)', repl, formula)

# --- Streamlit setup ---

st.set_page_config("Tc vs Electronegativity", layout="centered")
st.title("🧪 Assessing Chemical Composition for Superconducting Hydrides")
st.markdown("Data-Driven Modeling of Superconducting Hydrides: From Composition to Critical Parameters")

# --- Wczytanie CSV ---

try:
    df = pd.read_csv("Data-ternary.csv")
    # Zadbajmy, aby formuły w bazie były bez jedynek (na wszelki wypadek)
    df['formula_clean'] = df['formula'].apply(clean_formula)
except FileNotFoundError:
    st.error("❌ File `Data-ternary.csv` not found.")
    st.stop()

# --- Wprowadzanie wzoru użytkownika ---

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

# --- Wykres Tc vs Electronegativity ---

fig, ax = plt.subplots(figsize=(10, 6))

# Regiony na wykresie
ax.axvspan(2.00, 2.10, color='red', alpha=0.2, label="S1")
ax.axvspan(1.90, 2.00, color='orange', alpha=0.15, label="S2")
ax.axvspan(2.10, 2.20, color='orange', alpha=0.15)
ax.axvspan(1.80, 1.90, color='yellow', alpha=0.1, label="S3")
ax.axvspan(2.20, 2.30, color='yellow', alpha=0.1)

# Punkty z CSV
for _, row in df.iterrows():
    color = hf_to_color(row['hf'])
    marker = mass_to_marker(row['mass'])
    ax.scatter(row['en'], row['Tc'], color=color, marker=marker, edgecolor='black', s=70)

# Punkt nowego związku
if formula:
    ax.axvline(en, color='red', linestyle='--', linewidth=2, label=f"{formula} (new)")

ax.set_xlim(1.0, 3.2)
ax.set_ylim(0, 500)
ax.set_xlabel("Average Electronegativity")
ax.set_ylabel("Tc (K)")
ax.set_title("Tc vs Electronegativity with Hf and Mass Ratio")

# Legendy
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

line_legend = mlines.Line2D([], [], color='red', linestyle='--', label="Your compound")

region_legend = [
    mpatches.Patch(color='red', alpha=0.2, label='S1'),
    mpatches.Patch(color='orange', alpha=0.15, label='S2'),
    mpatches.Patch(color='yellow', alpha=0.1, label='S3')
]

left_legend = ax.legend(handles=region_legend + [line_legend], loc='upper left', fontsize=9, title="Region")
ax.add_artist(left_legend)

right_legend = ax.legend(
    handles=hf_legend + mass_legend + [line_legend],
    loc='upper right',
    fontsize=9,
    title="Legend"
)

st.pyplot(fig)

# --- Sekcja: Szukanie w pełnej przestrzeni kombinacji ---

st.markdown("---")
st.header("🔍 Search in All Possible Element Combinations")

col1, col2 = st.columns(2)
with col1:
    hf_min = st.number_input("Hf min", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    ratio_min = st.number_input("Mx/MH min", min_value=0.0, value=0.0, step=0.1)
    en_min = st.number_input("Electronegativity min", min_value=0.0, max_value=5.0, value=1.8, step=0.01)
with col2:
    hf_max = st.number_input("Hf max", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    ratio_max = st.number_input("Mx/MH max", min_value=0.0, value=50.0, step=0.1)
    en_max = st.number_input("Electronegativity max", min_value=0.0, max_value=5.0, value=2.3, step=0.01)

elements = [el for el in ELECTRONEGATIVITY.keys() if el != "H" and ELECTRONEGATIVITY[el] is not None]
x_values = [1, 2, 3]
y_values = [1, 2, 3]
z_values = [2, 4, 6, 8, 9, 10, 12]

compound_list = []

for A, B in itertools.combinations(elements, 2):
    for x, y, z in itertools.product(x_values, y_values, z_values):
        chi = (x * ELECTRONEGATIVITY[A] + y * ELECTRONEGATIVITY[B] + z * ELECTRONEGATIVITY["H"]) / (x + y + z)
        hf_val = z / (x + y + z)
        mr_val = (x * ATOMIC_MASSES[A] + y * ATOMIC_MASSES[B]) / (z * ATOMIC_MASSES["H"])

        if hf_min <= hf_val <= hf_max and ratio_min <= mr_val <= ratio_max and en_min <= chi <= en_max:
            formula_raw = f"{A}{x}{B}{y}H{z}"
            formula_clean = clean_formula(formula_raw)
            in_csv = formula_clean in df['formula_clean'].values
            compound_list.append((formula_clean, chi, hf_val, mr_val, "✅" if in_csv else "❌"))

# Stworzenie DataFrame
results_df = pd.DataFrame(compound_list, columns=["Formula", "Electronegativity", "Hf", "Mx/MH", "In CSV"])

# Podsumowanie
st.markdown(f"### 📊 Found {len(results_df)} possible compounds matching criteria")

if not results_df.empty:
    st.dataframe(results_df)
    st.download_button("Download CSV", results_df.to_csv(index=False), "possible_compounds.csv", "text/csv")
else:
    st.warning("No compounds match the selected criteria.")

# --- Logowanie (opcjonalne) w konsoli dla kontroli (możesz wyłączyć później) ---
in_csv_count = results_df['In CSV'].value_counts().get("✅", 0)
not_in_csv_count = results_df['In CSV'].value_counts().get("❌", 0)
total_combinations = sum(1 for _ in itertools.product(elements, repeat=2)) * len(x_values) * len(y_values) * len(z_values)

print(f"Total combinations (approx): {total_combinations}")
print(f"Filtered combinations: {len(results_df)}")
print(f"Compounds in CSV: {in_csv_count}")
print(f"Compounds NOT in CSV: {not_in_csv_count}")
