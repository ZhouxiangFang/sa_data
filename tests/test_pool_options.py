"""Pool options (--response_filter, --dedup) filter every stage's pools the same way."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calculate_diversity as cd
import helper
import select_data as sd
import train_jobs

DEFAULT = helper.DEFAULT_POOL
REFUSAL = helper.PoolOptions("refusal")
DEDUP = helper.PoolOptions(dedup=True)
BOTH = helper.PoolOptions("refusal", True)


class FakeTokenizer:
    """Formats a prompt as itself and counts characters as tokens."""

    chat_template = "fake"

    def apply_chat_template(self, conversations, tokenize, add_generation_prompt):
        return [conversation[0]["content"] for conversation in conversations]

    def __call__(self, texts, add_special_tokens, return_length, truncation):
        return {"length": [len(text) for text in texts]}


def wildguardmix_rows():
    """WildGuardMix-like rows: p1 and p3 have a refusal and a harmless compliance (p3's
    compliance first), p2 a refusal and a harmful compliance, p4 is a benign prompt and
    p5 has a second row with an empty response."""
    rows = [
        ("p1", "p1 refuses", "harmful", "unharmful", "refusal", ["a"]),
        ("p1", "p1 complies", "harmful", "unharmful", "compliance", ["a"]),
        ("p2", "p2 refuses", "harmful", "unharmful", "refusal", ["a", "b"]),
        ("p2", "p2 harms", "harmful", "harmful", "compliance", ["a", "b"]),
        ("p3", "p3 complies", "harmful", "unharmful", "compliance", ["b"]),
        ("p3", "p3 refuses", "harmful", "unharmful", "refusal", ["b"]),
        ("p4", "p4 answer", "unharmful", "unharmful", "compliance", ["benign"]),
        ("p5", "p5 refuses", "harmful", "unharmful", "refusal", ["b"]),
        ("p5", " ", "harmful", "unharmful", "refusal", ["b"]),
    ]
    df = pd.DataFrame(rows, columns=["prompt", "response", "prompt_harm_label",
                                     "response_harm_label", "response_refusal_label",
                                     "sub_abbrs"])
    names = {"a": "cat_a", "b": "cat_b", "benign": "benign"}
    df["subcategory"] = df["sub_abbrs"].map(lambda abbrs: ",".join(names[a] for a in abbrs))
    df["adversarial"] = [False, False, True, True, False, False, False, True, True]
    return df


class PoolOptionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        # Group sizes are counted from the data, not taken from the stats file.
        stats = {"wildguardmix": {
            name: {"abbr": abbr, "train": {"safe_response": 99}}
            for name, abbr in (("cat_a", "a"), ("cat_b", "b"), ("benign", "benign"))
        }}
        (self.root / "subcategory_stats.json").write_text(json.dumps(stats))
        self.df = wildguardmix_rows()
        load = lambda name, split: self.df.copy()
        for patch in (mock.patch.object(helper, "DATA_DIR", str(self.root)),
                      mock.patch.object(helper, "load_safety_dataset", load)):
            patch.start()
            self.addCleanup(patch.stop)
        helper._SUBCATEGORY_SIZES.clear()
        self.addCleanup(helper._SUBCATEGORY_SIZES.clear)

    def pool(self, abbr=None, options=DEFAULT, harmful_count=None):
        rows = (helper.eligible_rows(self.df, "wildguardmix", options=options) if abbr is None else
                helper.select_harmful_subcategory(self.df, "wildguardmix", abbr,
                                                  harmful_count, options))
        return rows.index.tolist()

    def test_tags_and_folders(self):
        self.assertEqual([o.tag for o in (DEFAULT, REFUSAL, DEDUP, BOTH)],
                         ["", "refusal", "dedup", "refusal_dedup"])
        self.assertEqual(REFUSAL.folder("wildguardmix"), "wildguardmix_refusal")
        self.assertEqual(DEFAULT.folder("wildguardmix"), "wildguardmix")
        self.assertEqual(helper.pool_options(argparse.Namespace()), DEFAULT)
        self.assertEqual(helper.pool_options(argparse.Namespace(response_filter="refusal",
                                                                dedup=True)), BOTH)
        with self.assertRaises(ValueError):
            helper.PoolOptions("compliance")

    def test_safe_responses_keep_compliances_and_repeated_prompts(self):
        self.assertEqual(self.pool(), [0, 1, 2, 4, 5, 7])

    def test_refusal_keeps_only_refusals(self):
        self.assertEqual(self.pool(options=REFUSAL), [0, 2, 5, 7])

    def test_dedup_keeps_each_prompts_first_eligible_row(self):
        # p3's first eligible row is its compliance; dedup does not prefer refusals.
        self.assertEqual(self.pool(options=DEDUP), [0, 2, 4, 7])
        self.assertEqual(self.pool(options=BOTH), [0, 2, 5, 7])

    def test_a_subcategory_without_eligible_rows_is_an_empty_pool(self):
        pool = helper.select_harmful_subcategory(self.df[self.df["prompt"] == "p4"],
                                                 "wildguardmix", "a")
        self.assertEqual((len(pool), list(pool.columns)), (0, list(self.df.columns)))

    def test_refusal_needs_refusal_labels(self):
        for dataset in ("aegis", "gretel"):
            with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                REFUSAL.check(dataset)
        with self.assertRaises(ValueError):
            helper.eligible_rows(self.df, "aegis", options=REFUSAL)
        with self.assertRaises(ValueError):
            self.pool("benign", REFUSAL)
        DEDUP.check("aegis")

    def test_subcategory_pools_are_rows_of_the_selection_pool(self):
        for options in (DEFAULT, REFUSAL, DEDUP, BOTH):
            whole = set(self.pool(options=options))
            for abbr in ("a", "b"):
                with self.subTest(options=options.tag, abbr=abbr):
                    self.assertLessEqual(set(self.pool(abbr, options)), whole)
        self.assertEqual(self.pool("b", DEDUP), [2, 4, 7])
        self.assertEqual(self.pool("b", REFUSAL), [2, 5, 7])

    def test_group_sizes_count_the_rows_that_pass(self):
        self.assertEqual(helper.group_sizes("wildguardmix"), {"a": 3, "b": 4, "benign": 1})
        self.assertEqual(helper.group_sizes("wildguardmix", options=DEDUP),
                         {"a": 2, "b": 3, "benign": 1})
        self.assertEqual(helper.group_sizes("wildguardmix", options=REFUSAL), {"a": 2, "b": 3})
        self.assertEqual(helper.training_groups("wildguardmix", 3), ["a", "b"])
        self.assertEqual(helper.training_groups("wildguardmix", 3, options=REFUSAL), ["b"])
        # A baseline pools the groups that pass with the same options.
        self.assertEqual(self.pool("random", REFUSAL, harmful_count=3), [2, 5, 7])
        self.assertEqual(self.pool("random", DEFAULT, harmful_count=3), [0, 1, 2, 4, 5, 7])

    def test_planner_uses_the_options(self):
        self.assertEqual(train_jobs.dataset_groups("wildguardmix", 3, "subcategory",
                                                   options=REFUSAL),
                         (["b", "random", "evenly"], {"a": 2}))
        self.assertEqual(train_jobs.dataset_groups("wildguardmix", 3, "subcategory"),
                         (["a", "b", "random", "evenly"], {}))
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            train_jobs.parse_args(["--ckpt_dir", "x", "--response_filter", "refusal",
                                   "--datasets", "aegis"])
        args = train_jobs.parse_args(["--ckpt_dir", "x", "--dedup"])
        self.assertEqual(args.options, DEDUP)

    def test_training_draws_from_the_filtered_pool(self):
        samples = self.root / "train_samples"
        for options, abbr in ((REFUSAL, "b"), (REFUSAL, "random"), (DEFAULT, "b")):
            with self.subTest(options=options.tag, abbr=abbr):
                args = argparse.Namespace(
                    num_train=3, harmful_rate=1.0, general_rate=0.0,
                    alignment_dataset="wildguardmix", abbr=abbr, group_by="subcategory",
                    benign_data_type="vanilla_benign", seed=7, max_length=100,
                    response_filter=options.response_filter, dedup=options.dedup)
                train_df, available, _, _ = helper.build_training_data(
                    args, FakeTokenizer(), samples_dir=samples)
                expected = self.df.loc[self.pool(abbr, options, harmful_count=3)]
                self.assertEqual(available, len(expected))
                self.assertLessEqual(set(train_df["response"]), set(expected["response"]))
                folder = options.folder("wildguardmix")
                self.assertEqual(train_df.attrs["sample_files"]["harmful"],
                                 samples / "seed7" / folder / f"{abbr}_3_len100.jsonl")
        refusals = helper.read_sample(samples / "seed7" / "wildguardmix_refusal" / "b_3_len100.jsonl")
        self.assertTrue(refusals["response"].str.endswith("refuses").all())

    def save_subset(self, rows, options, name="lexical_hdd"):
        path = helper.selection_csv(7, "wildguardmix", name, len(rows), options)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.loc[rows, ["prompt", "response"]].to_csv(path, index=False)
        return path

    def test_selected_subsets_follow_the_options(self):
        # Without dedup a prompt may come with each of its responses...
        self.save_subset([0, 1, 2], DEFAULT)
        self.assertEqual(len(helper.load_selection(7, "wildguardmix", "lexical_hdd", 3)), 3)
        self.assertEqual(helper.selection_names(7, "wildguardmix", 3), ["lexical_hdd"])
        # ...but each row only as often as the pool has it.
        self.save_subset([0, 0, 2], DEFAULT, "twice")
        with self.assertRaises(ValueError):
            helper.load_selection(7, "wildguardmix", "twice", 3)
        # The options' own folder; dedup refuses a repeated prompt, refusal a compliance.
        path = self.save_subset([0, 2, 5], REFUSAL)
        self.assertEqual(path.parent.name, "wildguardmix_refusal")
        self.assertEqual(len(helper.load_selection(7, "wildguardmix", "lexical_hdd", 3,
                                                   REFUSAL)), 3)
        self.save_subset([0, 1, 2], REFUSAL, "compliance")
        self.save_subset([0, 1, 2], DEDUP, "repeated")
        for options, name in ((REFUSAL, "compliance"), (DEDUP, "repeated")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                helper.load_selection(7, "wildguardmix", name, 3, options)
        self.assertEqual(train_jobs.dataset_groups("wildguardmix", 3, "selection", seed=7,
                                                   options=REFUSAL),
                         (["compliance", "lexical_hdd"], {}))

    def test_selection_pool_keeps_every_row_by_default(self):
        with mock.patch("builtins.print"):
            pool = sd.load_pool("wildguardmix", FakeTokenizer(), 100)
            deduped = sd.load_pool("wildguardmix", FakeTokenizer(), 100, DEDUP)
            refusals = sd.load_pool("wildguardmix", FakeTokenizer(), 100, REFUSAL)
        self.assertEqual(pool["pool_index"].tolist(), [0, 1, 2, 4, 5, 7])
        self.assertEqual(deduped["pool_index"].tolist(), [0, 2, 4, 7])
        self.assertEqual(refusals["pool_index"].tolist(), [0, 2, 5, 7])
        self.assertEqual(sd.output_path("out", 7, "wildguardmix", "lexical_hdd", None, 600,
                                        REFUSAL),
                         Path("out/seed7/wildguardmix_refusal/lexical_hdd_600.csv"))
        self.assertEqual(sd.output_path("out", 7, "wildguardmix", "lexical_hdd", None, 600),
                         Path("out/seed7/wildguardmix/lexical_hdd_600.csv"))
        # Training reads the subsets where select_data.py writes them.
        self.assertEqual(helper.selection_csv(7, "wildguardmix", "lexical_hdd", 600, REFUSAL),
                         sd.output_path(self.root, 7, "wildguardmix", "lexical_hdd", None, 600,
                                        REFUSAL))

    def test_scripts_accept_the_options(self):
        cases = {
            sd: [(["--response_filter", "refusal"], ["wildguardmix"]),
                 (["--dedup"], list(cd.DEFAULT_DATASETS))],
            cd: [(["--response_filter", "refusal"], ["wildguardmix"]),
                 (["--dedup"], list(cd.DEFAULT_DATASETS))],
        }
        for module, runs in cases.items():
            for argv, datasets in runs:
                with self.subTest(module=module.__name__, argv=argv), \
                        mock.patch.object(sys, "argv", ["prog", *argv]):
                    args = module.parse_args()
                    self.assertEqual(args.datasets, datasets)
            with self.subTest(module=module.__name__, argv="refusal on aegis"), \
                    mock.patch.object(sys, "argv", ["prog", "--response_filter", "refusal",
                                                    "--datasets", "aegis"]), \
                    mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                module.parse_args()
        # Diversity scores of other options get their own root.
        for argv, root in (([], "diversity"), (["--dedup"], "diversity/dedup"),
                           (["--response_filter", "refusal", "--dedup"],
                            "diversity/refusal_dedup")):
            with mock.patch.object(sys, "argv", ["prog", *argv]):
                self.assertTrue(cd.parse_args().output_dir.as_posix().endswith(root))
        with mock.patch.object(sys, "argv", ["prog", "--dedup", "--split", "test"]), \
                mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            cd.parse_args()

    def test_diversity_scores_the_training_pools(self):
        seen = {}

        def record(seed, tokenizer, max_length, *, harmful_pool, abbr, options, **kwargs):
            seen[abbr] = (harmful_pool.index.tolist(), options)
            raise ValueError("Requested more examples than the test pools hold")

        tags = lambda texts, num_proc=4: [(["NN"], [text]) for text in texts]
        for options in (DEFAULT, REFUSAL, DEDUP):
            seen.clear()
            with self.subTest(options=options.tag), \
                    mock.patch.object(cd, "sample_examples", record), \
                    mock.patch.object(cd, "tag_corpus", tags), \
                    mock.patch("builtins.print"), self.assertRaises(ValueError):
                cd.compute_dataset_diversity(self.df, "wildguardmix", tokenizer=FakeTokenizer(),
                                             num_train=3, n_clusters=1, knn_ks=(1,),
                                             options=options)
            expected = {"a": self.pool("a", options), "b": self.pool("b", options),
                        "random": self.pool("random", options, harmful_count=3),
                        "evenly": self.pool("evenly", options, harmful_count=3)}
            if options.response_filter == "safe":
                expected["benign"] = self.pool("benign", options)
            self.assertEqual({abbr: pool for abbr, (pool, _) in seen.items() if abbr != "overall"},
                             expected)
            self.assertEqual({used for _, used in seen.values()}, {options})


if __name__ == "__main__":
    unittest.main()
