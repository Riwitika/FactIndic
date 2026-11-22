#!/usr/bin/env python3
"""Independent reproducibility check script for docs/RESULTS_AND_DISCUSSION.md.

Verifies that all numerical values, ratios, macro averages, micro counts, and
latencies cited in docs/RESULTS_AND_DISCUSSION.md are 100% reproducible from
the machine-readable evaluation outputs in experiment_results.jsonl.
"""

import json
from pathlib import Path
from collections import defaultdict


def main():
    results_path = Path("data/benchmark_results/controlled_run/local_mdeberta/experiment_results.jsonl")
    assert results_path.exists(), f"Missing {results_path}"

    with open(results_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) == 60, f"Expected 60 records, found {len(records)}"

    # Group by condition and language
    cond_groups = defaultdict(list)
    for r in records:
        cond_groups[(r["condition_id"], r["language"])].append(r)
        cond_groups[(r["condition_id"], "all")].append(r)

    # Expected values from Table 1
    # Key: (condition_id, language): (eval_n, pooled_items, macro_mean_entail, macro_entail_ratio, macro_contra_ratio, macro_neutral_ratio, micro_entail_ratio, micro_contra_ratio, mean_lat_ms)
    expected_table_1 = {
        ("C1", "en"): (5, 5, 0.0547, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 4072.41),
        ("C1", "hi"): (5, 5, 0.2732, 0.2000, 0.0000, 0.8000, 0.2000, 0.0000, 2118.47),
        ("C1", "all"): (10, 10, 0.1639, 0.1000, 0.0000, 0.9000, 0.1000, 0.0000, 3095.44),
        ("C2", "en"): (5, 5, 0.4549, 0.4000, 0.0000, 0.6000, 0.4000, 0.0000, 2035.65),
        ("C2", "hi"): (5, 5, 0.2543, 0.2000, 0.0000, 0.8000, 0.2000, 0.0000, 2058.92),
        ("C2", "all"): (10, 10, 0.3546, 0.3000, 0.0000, 0.7000, 0.3000, 0.0000, 2047.29),
        ("C3", "en"): (5, 54, 0.4836, 0.5440, 0.0654, 0.3906, 0.5741, 0.0556, 15404.81),
        ("C3", "hi"): (5, 40, 0.6481, 0.6590, 0.1022, 0.2387, 0.7000, 0.0750, 14334.90),
        ("C3", "all"): (10, 94, 0.5658, 0.6015, 0.0838, 0.3147, 0.6277, 0.0638, 14869.85),
        ("C4", "en"): (5, 39, 0.4331, 0.4250, 0.0625, 0.5125, 0.5385, 0.0769, 9036.29),
        ("C4", "hi"): (5, 67, 0.6771, 0.8025, 0.0550, 0.1425, 0.7910, 0.0597, 23748.60),
        ("C4", "all"): (10, 106, 0.5551, 0.6138, 0.0588, 0.3275, 0.6981, 0.0660, 16392.44),
        ("C5", "en"): (5, 63, 0.5700, 0.5518, 0.0154, 0.4328, 0.5714, 0.0159, 9108.23),
        ("C5", "hi"): (5, 21, 0.5317, 0.6067, 0.1067, 0.2867, 0.6190, 0.0952, 4681.71),
        ("C5", "all"): (10, 84, 0.5509, 0.5792, 0.0610, 0.3598, 0.5833, 0.0357, 6894.97),
        ("C6", "en"): (5, 43, 0.5192, 0.4809, 0.0533, 0.4657, 0.5116, 0.0465, 5762.58),
        ("C6", "hi"): (5, 57, 0.3268, 0.3718, 0.0286, 0.5996, 0.1754, 0.0351, 1527.92),
        ("C6", "all"): (10, 100, 0.4230, 0.4264, 0.0410, 0.5327, 0.3200, 0.0400, 3645.25),
    }

    print("Verifying all Table 1 numbers against experiment_results.jsonl...")

    for key, expected in expected_table_1.items():
        cond_id, lang = key
        recs = cond_groups[key]
        n = len(recs)
        strat = recs[0]["verification_strategy"]
        lat = sum(r["verification_latency_ms"] for r in recs) / n

        if strat == "whole_document":
            tot_items = n
            ent_cnt = sum(1 for r in recs if r["aggregated_metrics"]["predicted_label"] == "entailment")
            neut_cnt = sum(1 for r in recs if r["aggregated_metrics"]["predicted_label"] == "neutral")
            contra_cnt = sum(1 for r in recs if r["aggregated_metrics"]["predicted_label"] == "contradiction")
            macro_mean_ent = sum(r["aggregated_metrics"]["entailment_prob"] for r in recs) / n
            macro_ent = ent_cnt / n
            macro_neut = neut_cnt / n
            macro_contra = contra_cnt / n
            micro_ent = macro_ent
            micro_contra = macro_contra
        elif strat == "sentence_level":
            tot_items = sum(r["aggregated_metrics"]["total_sentences"] for r in recs)
            ent_items = sum(r["aggregated_metrics"]["entailed_sentences"] for r in recs)
            contra_items = sum(r["aggregated_metrics"]["contradicted_sentences"] for r in recs)
            macro_ent = sum(r["aggregated_metrics"]["entailment_ratio"] for r in recs) / n
            macro_neut = sum(r["aggregated_metrics"]["neutral_ratio"] for r in recs) / n
            macro_contra = sum(r["aggregated_metrics"]["contradiction_ratio"] for r in recs) / n
            macro_mean_ent = sum(r["aggregated_metrics"]["mean_sentence_entailment"] for r in recs) / n
            micro_ent = ent_items / tot_items
            micro_contra = contra_items / tot_items
        elif strat == "hybrid_claim_level":
            tot_items = sum(r["aggregated_metrics"]["total_claims"] for r in recs)
            ent_items = sum(r["aggregated_metrics"]["entailed_claims"] for r in recs)
            contra_items = sum(r["aggregated_metrics"]["contradicted_claims"] for r in recs)
            macro_ent = sum(r["aggregated_metrics"]["entailment_ratio"] for r in recs) / n
            macro_neut = sum(r["aggregated_metrics"]["neutral_ratio"] for r in recs) / n
            macro_contra = sum(r["aggregated_metrics"]["contradiction_ratio"] for r in recs) / n
            macro_mean_ent = sum(r["aggregated_metrics"]["mean_claim_entailment"] for r in recs) / n
            micro_ent = ent_items / tot_items
            micro_contra = contra_items / tot_items

        # Compare with expected
        exp_n, exp_items, exp_mean_ent, exp_macro_ent, exp_macro_contra, exp_macro_neut, exp_micro_ent, exp_micro_contra, exp_lat = expected
        assert n == exp_n, f"{key} N mismatch: {n} vs {exp_n}"
        assert tot_items == exp_items, f"{key} Items mismatch: {tot_items} vs {exp_items}"
        assert abs(macro_mean_ent - exp_mean_ent) < 0.0001, f"{key} MacroMeanEnt mismatch: {macro_mean_ent} vs {exp_mean_ent}"
        assert abs(macro_ent - exp_macro_ent) < 0.0001, f"{key} MacroEnt mismatch: {macro_ent} vs {exp_macro_ent}"
        assert abs(macro_contra - exp_macro_contra) < 0.0001, f"{key} MacroContra mismatch: {macro_contra} vs {exp_macro_contra}"
        assert abs(macro_neut - exp_macro_neut) < 0.0001, f"{key} MacroNeut mismatch: {macro_neut} vs {exp_macro_neut}"
        assert abs(micro_ent - exp_micro_ent) < 0.0001, f"{key} MicroEnt mismatch: {micro_ent} vs {exp_micro_ent}"
        assert abs(micro_contra - exp_micro_contra) < 0.0001, f"{key} MicroContra mismatch: {micro_contra} vs {exp_micro_contra}"
        assert abs(lat - exp_lat) < 0.05, f"{key} Latency mismatch: {lat} vs {exp_lat}"

        print(f"  [PASS] {cond_id:<3} {lang:<4}: N={n}, Items={tot_items:<3}, MacroEnt={macro_ent:.4f}, MacroContra={macro_contra:.4f}, MeanEntScore={macro_mean_ent:.4f}, Lat={lat:.2f}ms")

    print("\nALL 18 TABLE ROWS (AND DERIVED SUB-CONDITIONS) 100% REPRODUCED AND VERIFIED!")


if __name__ == "__main__":
    main()
