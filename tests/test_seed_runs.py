"""Seeded runs: training and diversity draw the same examples, and correlations pair them."""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calculate_diversity as cd
import eval_if
import eval_safe
import helper
import plot_bars_diversity as pbd


class FakeTokenizer:
    """Formats a prompt as itself and counts characters as tokens."""

    def apply_chat_template(self, conversations, tokenize, add_generation_prompt):
        return [conversation[0]["content"] for conversation in conversations]

    def __call__(self, texts, add_special_tokens, return_length, truncation):
        return {"length": [len(text) for text in texts]}


class FakeEmbedder:
    """Deterministic, distinct unit vectors per text."""

    model_id = "fake"

    def encode(self, seqs):
        return self.encode_texts([" ".join(seq) for seq in seqs])

    def encode_texts(self, texts):
        vectors = [np.random.default_rng(sum(map(ord, text))).normal(size=8) for text in texts]
        return np.asarray([vector / np.linalg.norm(vector) for vector in vectors])


def gretel_rows():
    rows = []
    for tactic, count in (("quid pro quo", 12), ("phishing", 10), ("blackmail", 3)):
        for i in range(count):
            rows.append({"prompt": f"{tactic} prompt number {i} asks for help",
                         "response": "no", "prompt_harm_label": "harmful",
                         "response_harm_label": "safe", "sub_abbrs": ["malicious_use"],
                         "tactic": tactic})
    return pd.DataFrame(rows)


def pairs(prefix, count):
    return pd.DataFrame({"prompt": [f"{prefix} {i}" for i in range(count)],
                         "response": ["ok"] * count})


class SeedTests(unittest.TestCase):
    def test_run_seeds_are_multiples_of_42(self):
        self.assertEqual(helper.run_seeds(), [42, 84, 126])
        self.assertEqual(helper.run_seeds(4), [42, 84, 126, 168])
        self.assertEqual(helper.seed_suffix(84), "_seed84")
        self.assertEqual(helper.insert_seed_dir("results/if_eval/", 84),
                         Path("results/seed84/if_eval"))
        self.assertEqual(helper.seed_diversity_csv("diversity/bge-m3/csv/a.csv", 42),
                         Path("diversity/seed42/bge-m3/csv/a.csv"))
        with self.assertRaises(ValueError):
            helper.run_seeds(0)

    def test_seeded_checkpoints_get_seed_result_folders(self):
        safe = eval_safe.result_location("R", "olmo2_git20k_1800_aegis_hate_seed84")
        self.assertEqual(safe, ("R/seed84/olmo2_git20k_1800_aegis/aegis_hate", "aegis_hate"))
        # Checkpoints named before seeds were added are seed 42.
        legacy = eval_safe.result_location("R", "olmo2_git20k_1800_aegis_hate")
        self.assertEqual(legacy[0], "R/seed42/olmo2_git20k_1800_aegis/aegis_hate")
        ifeval = eval_if.result_location("R/if_eval", "olmo2_git20k_1800_aegis_hate_seed84")
        self.assertEqual(ifeval, ("R/seed84/if_eval/olmo2_git20k_1800_aegis", "aegis_hate"))
        # Instruct baselines are not seeded runs.
        self.assertEqual(eval_safe.result_location("R", "olmo2-ins")[0], "R/olmo2-ins")
        self.assertEqual(eval_if.result_location("R/if_eval", "olmo2-ins")[0], "R/if_eval")

    def test_diversity_scores_exactly_the_harmful_examples_of_each_training_run(self):
        df = gretel_rows()
        benign, general = pairs("benign", 20), pairs("general", 20)
        harmful_count = 4
        drawn = {}
        real = cd.sample_examples

        def record(seed, *args, **kwargs):
            parts, info = real(seed, *args, **kwargs)
            drawn[kwargs["abbr"], seed] = sorted(parts["harmful"]["prompt"])
            return parts, info

        tags = lambda texts, num_proc: [(text.split(), text.lower().split()) for text in texts]
        with mock.patch.object(cd, "sample_examples", record), \
                mock.patch.object(cd, "tag_corpus", tags):
            by_seed = cd.compute_dataset_diversity(
                df, "gretel", tokenizer=FakeTokenizer(), num_train=harmful_count,
                group_by="tactic", n_clusters=2, knn_ks=(1,), num_proc=1,
                embedders={"fake": FakeEmbedder()},
            )["fake"]
        self.assertEqual(sorted(set(by_seed["seed"])), [42, 84, 126])
        self.assertEqual(sorted(set(by_seed["abbr"])),
                         ["evenly", "overall", "phishing", "quid_pro_quo", "random"])

        for abbr in ("quid_pro_quo", "phishing", "random", "evenly"):
            pool = helper.select_tactic(df, "gretel", abbr, harmful_count)
            for seed in helper.run_seeds():
                # align.py's 1:1:1 mixture with the same harmful count and seed.
                args = argparse.Namespace(
                    num_train=3 * harmful_count, harmful_rate=1 / 3, general_rate=1 / 3,
                    alignment_dataset="gretel", abbr=abbr, group_by="tactic",
                    benign_data_type="vanilla_benign", seed=seed, max_length=1000)
                train_df, _, _, _ = helper.build_training_data(
                    args, FakeTokenizer(), harmful_pool=pool, benign_pool=benign,
                    general_pool=general)
                trained = sorted(set(train_df["prompt"]) & set(pool["prompt"]))
                self.assertEqual(drawn[abbr, seed], trained, (abbr, seed))
        # Different seeds train on different samples.
        self.assertNotEqual(drawn["quid_pro_quo", 42], drawn["quid_pro_quo", 84])


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        metrics = dict.fromkeys(pbd.METRICS, 0.0)
        rows = []
        for abbr, base in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
            for seed, offset in ((42, 0.0), (84, 0.5)):
                rows.append({"abbr": abbr, "seed": seed, **metrics,
                             "lexical_hdd↑": base + offset})
        self.diversity = pd.DataFrame(rows)

    def write_run(self, root, seed, folder, abbr, score):
        run = root / f"seed{seed}" / folder / f"aegis_{abbr}"
        run.mkdir(parents=True)
        pd.DataFrame({"dataset": ["Avg"], "harmful_score_pct": [score]}).to_csv(
            run / f"aegis_{abbr}_overall.csv", index=False)

    def test_runs_pair_with_their_own_seed_and_points_average_the_same_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model, shift in (("m1", 0.0), ("m2", 10.0)):
                for abbr, score in (("a", 30.0), ("b", 20.0), ("c", 10.0)):
                    self.write_run(root, 42, f"{model}_1800_aegis", abbr, score + shift)
                    # m2 has no seed-84 run of c.
                    if not (model == "m2" and abbr == "c"):
                        self.write_run(root, 84, f"{model}_1800_aegis", abbr, score + shift + 2)
            safety = pbd.load_safety(root, "aegis", 1800, {"a", "b", "c"}, "1800_aegis",
                                     seeds=[42, 84])
        self.assertEqual(len(safety), 11)
        points, runs = pbd.join_scores(self.diversity, safety)
        run = runs.query("model == 'm1' and abbr == 'a' and seed == 84").iloc[0]
        self.assertEqual(run["lexical_hdd↑"], 1.5)
        self.assertEqual(run["harmful_score_pct"], 32.0)

        point = points.query("model == 'm1' and abbr == 'a'").iloc[0]
        self.assertEqual((point["harmful_score_pct"], point["lexical_hdd↑"]), (31.0, 1.25))
        self.assertEqual((point["n_seeds"], point["seeds"]), (2, "42 84"))
        # m2/c has only seed 42, so its diversity is seed 42's sample alone.
        point = points.query("model == 'm2' and abbr == 'c'").iloc[0]
        self.assertEqual((point["harmful_score_pct"], point["lexical_hdd↑"]), (20.0, 3.0))
        # The average needs every model: c uses only seed 42.
        average = points.query("model == 'Average' and abbr == 'c'").iloc[0]
        self.assertEqual((average["harmful_score_pct"], average["lexical_hdd↑"],
                          average["n_seeds"]), (15.0, 3.0, 1))

        points, runs = (pbd.mark_excluded(frame, set()) for frame in (points, runs))
        by_seed = pbd.seed_correlations(runs, ["pearson"])
        self.assertEqual(sorted(by_seed["seed"].unique()), [42, 84])
        table = pbd.add_seed_summary(pbd.correlation_table(points, ["pearson"]), by_seed)
        row = table.query("model == 'm1' and metric == 'lexical_hdd↑'").iloc[0]
        self.assertAlmostEqual(row["coefficient"], -1.0)
        self.assertEqual((row["n_subcategories"], row["n_runs"], row["n_seed_correlations"]),
                         (3, 6, 2))

    def test_load_diversity_reads_the_seed_folders_listed_by_the_mean_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mean = root / "bge-m3" / "csv" / "aegis_train_diversity_600_1.0.csv"
            mean.parent.mkdir(parents=True)
            for seed, rows in self.diversity.groupby("seed"):
                path = root / f"seed{seed}" / "bge-m3" / "csv" / mean.name
                path.parent.mkdir(parents=True)
                rows.to_csv(path, index=False)
            cd.seed_means(self.diversity).to_csv(mean, index=False)
            loaded = pbd.load_diversity(mean)
            self.assertEqual(sorted(loaded["seed"].unique()), [42, 84])
            row = loaded.query("abbr == 'b' and seed == 84").iloc[0]
            self.assertEqual(row["lexical_hdd↑"], 2.5)
            # A mean CSV without seeds predates seed folders.
            cd.seed_means(self.diversity).drop(columns="seeds").to_csv(mean, index=False)
            with self.assertRaises(ValueError):
                pbd.load_diversity(mean)


if __name__ == "__main__":
    unittest.main()
