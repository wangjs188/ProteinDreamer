import csv
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def read_table(name, numeric):
    with (ROOT / "data" / name).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["pos"] = int(row["pos"])
        for key in numeric:
            row[key] = float(row[key])
            if not isfinite(row[key]):
                raise ValueError(f"Non-finite input: {name}, position {row['pos']}, {key}")
    return rows


def write_table(name, rows, columns):
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    with (output / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def top_ten(rows, key):
    """Include ties at the tenth score, equivalent to minimum rank <= 10."""
    if not rows:
        return set()
    cutoff = sorted((row[key] for row in rows), reverse=True)[min(10, len(rows)) - 1]
    return {row["mutant"] for row in rows if row[key] >= cutoff}


def main():
    sites = read_table("sites.csv", ["distance_max_A"])
    numeric = ["foldx_median", "foldx_iqr", "esm2_llr", "pssm", "pd_score",
               "saprot_score", "dlkcat_RP", "dlkcat_native_reference"]
    mutations = read_table("substitutions.csv", numeric)
    if len({row["pos"] for row in sites}) != len(sites):
        raise ValueError("Duplicate site positions")
    if len({row["mutant"] for row in mutations}) != len(mutations):
        raise ValueError("Duplicate mutation identifiers")
    by_site = defaultdict(list)
    for row in mutations:
        row["dlkcat_delta"] = row["dlkcat_RP"] - row["dlkcat_native_reference"]
        by_site[row["pos"]].append(row)
    if set(by_site) != {site["pos"] for site in sites}:
        raise ValueError("Site and substitution tables cover different positions")

    retained, decisions = [], []
    for site in sorted(sites, key=lambda row: row["pos"]):
        group = by_site[site["pos"]]
        alternatives = set("ACDEFGHIKLMNPQRSTVWY") - {site["wt"]}
        if len(group) != 19 or {row["mt"] for row in group} != alternatives:
            raise ValueError(f"Expected 19 non-native alternatives at {site['pos']}")
        if any(row["mutant"] != f"{site['wt']}{site['pos']}{row['mt']}" for row in group):
            raise ValueError(f"Mutation identity mismatch at {site['pos']}")


        absolute = [abs(row["foldx_median"]) for row in group]
        e, h = median(absolute), max(absolute)
        n = sum(value >= 0.5 for value in absolute)
        c = sum(row["foldx_iqr"] <= 0.8 for row in group) / 19
        profiles = [name for name, passed in [
            ("hotspot_effect", h >= 1.0 or n >= 2),
            ("robust_moderate", e >= 0.015 and h >= 0.25 and c >= 0.8),
            ("neutral_control", e <= 0.01 and h <= 0.05 and c >= 0.9),
            ("context_sensitive", e <= 0.02 and c < 0.3),
        ] if passed]
        structural = site["distance_max_A"] <= 8.0
        foldx = structural and bool(profiles)

       
        esm_max = max(row["esm2_llr"] for row in group)
        pssm_max = max(row["pssm"] for row in group)
        evolution = foldx and esm_max > -2 and pssm_max > -3
        clean = [row for row in group if evolution and not (
            row["cupsat_stability"] == "Destabilising" and
            row["cupsat_torsion"] == "Unfavourable")]
        retained.extend(clean)
        decisions.append(dict(site=site["wt"] + str(site["pos"]), pos=site["pos"],
            distance_max_A=site["distance_max_A"], foldx_E=e, foldx_H=h, foldx_N=n,
            foldx_C=c, foldx_profile="|".join(profiles) or "other",
            pass_structure=structural, pass_foldx=foldx, esm2_max=esm_max,
            pssm_max=pssm_max, pass_evolution=evolution, retained_substitutions=len(clean),
            selected=False, first_exclusion=("structure" if not structural else
            "FoldX" if not foldx else "ESM2/PSSM" if not evolution else
            "CUPSAT" if not clean else "")))

    
    pd_top, dlk_top = top_ten(retained, "pd_score"), top_ten(retained, "dlkcat_delta")
    clean_by_site = defaultdict(list)
    for row in retained:
        clean_by_site[row["pos"]].append(row)
    for site in decisions:
        group = clean_by_site.get(site["pos"], [])
        if not group:
            continue
        positive = [row["pd_score"] for row in group if row["pd_score"] > 0]
        pd_median = median(positive) if positive else float("nan")
        saprot_max = max(row["saprot_score"] for row in group)
        a = sum(row["pd_score"] > 0 and row["dlkcat_delta"] > 0 for row in group)
        u = sum(row["mutant"] in pd_top | dlk_top for row in group)
        p = sum(row["mutant"] in pd_top for row in group)
        support = a + u + 2 * p
        selected = pd_median >= 0.99 and support >= 10 and saprot_max > 0
        reason = ("PD" if not pd_median >= 0.99 else "support" if support < 10 else
                  "SaProt" if saprot_max <= 0 else "")
        site.update(pd_positive_median=pd_median, saprot_max=saprot_max,
                    A=a, U=u, P=p, support=support, selected=selected, first_exclusion=reason)

    chosen = [site for site in decisions if site["selected"]]
    positions = {site["pos"] for site in chosen}
    chosen_mutations = [row for row in retained if row["pos"] in positions]
    counts = [
        dict(stage="Structural candidates", sites=sum(s["pass_structure"] for s in decisions),
             substitutions=19 * sum(s["pass_structure"] for s in decisions)),
        dict(stage="After FoldX", sites=sum(s["pass_foldx"] for s in decisions),
             substitutions=19 * sum(s["pass_foldx"] for s in decisions)),
        dict(stage="After risk filtering", sites=len(clean_by_site),
             substitutions=len(retained)),
        dict(stage="Selected sites", sites=len(chosen), substitutions=len(chosen_mutations)),
    ]
    columns = list(decisions[0]) + [key for key in ["pd_positive_median", "saprot_max",
               "A", "U", "P", "support"] if key not in decisions[0]]
    write_table("site_decisions.csv", decisions, columns)
    write_table("selected_sites.csv", chosen,
                ["site", "retained_substitutions", "pd_positive_median", "saprot_max", "A", "U", "P", "support"])
    write_table("selected_substitutions.csv", chosen_mutations, list(mutations[0]))
    write_table("stage_counts.csv", counts, ["stage", "sites", "substitutions"])
    for row in counts:
        print(f"{row['stage']}: {row['sites']} sites, {row['substitutions']} substitutions")
    print("Selected:", ", ".join(site["site"] for site in chosen))


if __name__ == "__main__":
    main()
