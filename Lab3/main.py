import sys, pandas as pd, numpy as np
from scipy import stats
from difflib import get_close_matches

fn = sys.argv[1] if len(sys.argv)>1 else "owid-covid-data.csv"
print("Loading:", fn)
df = pd.read_csv(fn, low_memory=False)
print("Columns:", list(df.columns))

# parse date & numeric fields
date_col = next((c for c in ("date","zvit_date","report_date") if c in df.columns), "date")
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
for c in ["total_cases","new_cases","total_tests","new_tests"]:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

# 2. convert up to 4 repeated-value object cols to category
obj = df.select_dtypes(include=["object"]).columns.tolist()
n = len(df)
cand = [(c, df[c].nunique(dropna=True)) for c in obj]
cand = sorted([x for x in cand if x[1] <= max(50,0.5*n)], key=lambda x:x[1])[:4]
cats = [c for c,_ in cand]
for c in cats: df[c]=df[c].astype("category")
print("\n[2] Converted to category:", cats)

# 3. value_counts for 'continent' and 'test_units' (if present)
print("\n[3] value_counts -> to_frame:")
for key in ("continent","test_units","tests_units"):
    if key in df.columns:
        vc = df[key].value_counts(dropna=False)
        print(f"\n{key}:")
        print(vc.to_frame(name="count"))
    else:
        print(f"\n{key} — not present")

# 4. total cases per category (sum total_cases)
cases_col = "total_cases" if "total_cases" in df.columns else ("new_cases" if "new_cases" in df.columns else None)
if not cases_col:
    raise SystemExit("No total_cases/new_cases in dataset")
print(f"\n[4] Aggregation using '{cases_col}':")
pd.options.display.float_format = "{:,.2f}".format
for c in df.select_dtypes(include=["category"]).columns:
    agg = df.groupby(c, observed=False)[cases_col].sum().sort_values(ascending=False)
    print(f"\n- {c} (top 8):")
    print(agg.head(8).to_frame(name="total_"+cases_col))

# 5. pivot: date x country
country_col = next((c for c in ("location","country","iso_code") if c in df.columns), "location")
if country_col not in df.columns: raise SystemExit("No country/location column")
print(f"\n[5] Pivot by date x {country_col} using '{cases_col}' (show head).")
pivot = pd.pivot_table(df, values=cases_col, index=date_col, columns=country_col, aggfunc="sum")
pivot = pivot.sort_index()
print("Pivot shape:", pivot.shape)
print(pivot.head(6))

# 6. drop missing rows (any)
pivot_clean = pivot.dropna(how="any")
print("\n[6] After dropna(how='any') shape:", pivot.shape, "->", pivot_clean.shape)
print("Pivot_clean head:")
print(pivot_clean.head(5))

# helper: robust country matching
all_countries = list(pivot.columns)
def match(name):
    if name in all_countries: return name
    L = [c for c in all_countries if c.lower()==name.lower()]
    if L: return L[0]
    m = get_close_matches(name, all_countries, n=1, cutoff=0.7)
    return m[0] if m else None

# 7. Correlation test: Poland vs (Hungary, Czechia, Slovakia)
base = "Poland"
compares = ["Hungary","Czechia","Slovakia"]
base_m = match(base)
print(f"\n[7] Correlation tests: base={base} -> matched: {base_m}")
if not base_m:
    print(" Base country not found in dataset")
else:
    for c in compares:
        cm = match(c)
        if not cm:
            print(" Compare", c, "-> not found")
            continue
        s1 = pivot_clean[base_m]; s2 = pivot_clean[cm]
        # choose Pearson if both approx normal
        use_pearson = False
        try:
            if len(s1)<=5000 and len(s2)<=5000:
                p1 = stats.shapiro(s1)[1]; p2 = stats.shapiro(s2)[1]
                use_pearson = (p1>0.05 and p2>0.05)
            else:
                p1 = stats.normaltest(s1)[1]; p2 = stats.normaltest(s2)[1]
                use_pearson = (p1>0.05 and p2>0.05)
        except Exception:
            use_pearson = False
        if use_pearson:
            r,p = stats.pearsonr(s1,s2); method="Pearson"
        else:
            r,p = stats.spearmanr(s1,s2); method="Spearman"
        print(f" {base_m} vs {cm} -> method={method}, r={r:.4f}, p={p:.4e}")
        print("  sample pairs (first 6):")
        print(pivot_clean[[base_m,cm]].head(6))

# 8. Forecasting for groups (list from assignment). use cumulative total_cases (already cumulative).
groups = {
 "scandinavia":["Sweden","Norway","Denmark","Finland"],
 "benelux":["Belgium","Netherlands","Luxembourg"],
 "eu4":["Poland","Hungary","Czechia","Slovakia"],
 "arabian":["Saudi Arabia","Yemen","Oman","United Arab Emirates"],
 "asia_rok_hk_sg_tw":["South Korea","Hong Kong","Singapore","Taiwan"],
 "mong_china_viet":["Mongolia","China","Vietnam"],
 "north_africa_4":["Morocco","Algeria","Tunisia","Libya"],
 "south_africa_4":["South Africa","Namibia","Botswana","Lesotho"],
 "india_nepal_pk":["India","Nepal","Pakistan"],
 "south_america_4":["Brazil","Argentina","Colombia","Chile"],
 "baltic_4":["Estonia","Latvia","Lithuania","Lithuania"], # adjust if needed
 "balkan_4":["Slovenia","Croatia","Serbia","Bosnia and Herzegovina"],
 "pol_hun_cze_svk":["Poland","Hungary","Czechia","Slovakia"],
 "svn_hrv_ltu_lva":["Slovenia","Croatia","Lithuania","Latvia"],
 "bul_rom_ukr_alb":["Bulgaria","Romania","Ukraine","Albania"],
 "nafta":["Canada","United States","Mexico"]
}

print("\n[8] Forecasting (14 days) for groups — history tail (3) + forecast head (5)")
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    hw=True
except Exception:
    hw=False
for g, lst in groups.items():
    matched = [match(x) for x in lst]
    present = [m for m in matched if m]
    if not present:
        print(f"\n{g}: no countries found in dataset")
        continue
    series = pivot[present].sum(axis=1).asfreq("D", fill_value=np.nan).ffill().fillna(0)
    print(f"\nGroup {g} -> present: {present}")
    print(" History last 3:", list(series.iloc[-3:]))
    steps=14
    if hw and series.dropna().shape[0] > 5:
        try:
            fit = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated").fit()
            f = fit.forecast(steps)
        except Exception:
            f = pd.Series([series.iloc[-1]]*steps, index=pd.date_range(series.index[-1]+pd.Timedelta(days=1), periods=steps))
    else:
        f = pd.Series([series.iloc[-1]]*steps, index=pd.date_range(series.index[-1]+pd.Timedelta(days=1), periods=steps))
    print(" Forecast next 5:", list(map(float, f.iloc[:5])))
print("\n--- DONE ---")

