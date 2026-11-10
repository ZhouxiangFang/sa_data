import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import helper
import train_jobs


class FakeTokenizer:
    """Formats a prompt as itself and counts characters as tokens."""

    def apply_chat_template(self, conversations, tokenize, add_generation_prompt):
        return [conversation[0]["content"] for conversation in conversations]

    def __call__(self, texts, add_special_tokens, return_length, truncation):
        return {"length": [len(text) for text in texts]}


def pool_rows():
    # Prompts pandas would read back as missing values, and one with a line break.
    prompts = ["None", "N/A", "null", "two\nlines, with a comma", *(f"prompt {i}" for i in range(8))]
    return pd.DataFrame({"prompt": prompts, "response": [f"no {i}" for i in range(len(prompts))]})


class SelectionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.pool = pool_rows()
        for patch in (mock.patch.object(helper, "DATA_DIR", str(self.root)),
                      mock.patch.object(helper, "load_harmful_subcategory",
                                        lambda dataset, abbr, count=None: self.pool)):
            patch.start()
            self.addCleanup(patch.stop)

    def save(self, name, rows, seed=7, dataset="aegis"):
        path = helper.selection_csv(seed, dataset, name, len(rows))
        path.parent.mkdir(parents=True, exist_ok=True)
        # The columns select_data.py writes around the prompt and response.
        rows.assign(pool_index=range(len(rows)), order=0, score=0.5).to_csv(path, index=False)
        return path

    def test_subset_is_read_back_exactly(self):
        subset = self.pool.iloc[[0, 1, 2, 3, 6, 9]]
        path = self.save("semantic_vendi_bge-m3", subset)
        self.assertEqual(path, self.root / "seed7" / "aegis" / "semantic_vendi_bge-m3_6.csv")
        loaded = helper.load_selection(7, "aegis", "semantic_vendi_bge-m3", 6)
        self.assertEqual(loaded.values.tolist(), subset.values.tolist())
        self.assertEqual(helper.selection_names(7, "aegis", 6), ["semantic_vendi_bge-m3"])

    def test_wrong_subsets_are_refused(self):
        self.save("short", self.pool.iloc[:5])
        self.save("foreign", pd.concat([self.pool.iloc[:5], pd.DataFrame(
            {"prompt": ["not in the pool"], "response": ["no"]})]))
        self.save("repeated", self.pool.iloc[[0, 1, 2, 3, 4, 4]])
        for name, count in (("missing", 6), ("short", 6), ("foreign", 6), ("repeated", 6)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                helper.load_selection(7, "aegis", name, count)

    def test_training_mixture_uses_the_whole_subset(self):
        subset = self.pool.iloc[[0, 1, 2, 3, 6, 9]]
        self.save("lexical_hdd", subset)
        args = argparse.Namespace(num_train=6, harmful_rate=1.0, general_rate=0.0,
                                  alignment_dataset="aegis", abbr="lexical_hdd",
                                  group_by="selection", benign_data_type="vanilla_benign",
                                  seed=7, max_length=100)
        samples = self.root / "train_samples"
        train_df, available, _, overlong = helper.build_training_data(
            args, FakeTokenizer(), samples_dir=samples)
        self.assertEqual((available, overlong), (6, 0))
        self.assertEqual(sorted(map(tuple, train_df[["prompt", "response"]].values)),
                         sorted(map(tuple, subset.values)))
        # The run's sample file is kept apart from the subcategory samples.
        sample_file = samples / "seed7" / "aegis_selection" / "lexical_hdd_6_len100.jsonl"
        self.assertEqual(train_df.attrs["sample_files"]["harmful"], sample_file)
        again, *_ = helper.build_training_data(args, FakeTokenizer(), samples_dir=samples)
        self.assertEqual(again.values.tolist(), train_df.values.tolist())
        # A subset selected again no longer matches the saved sample.
        self.save("lexical_hdd", self.pool.iloc[[0, 1, 2, 3, 6, 10]])
        with self.assertRaises(ValueError):
            helper.build_training_data(args, FakeTokenizer(), samples_dir=samples)

    def test_overlong_prompt_is_refused(self):
        self.pool.loc[5, "prompt"] = "x" * 200
        self.save("lexical_hdd", self.pool.iloc[:6])
        args = argparse.Namespace(num_train=6, harmful_rate=1.0, general_rate=0.0,
                                  alignment_dataset="aegis", abbr="lexical_hdd",
                                  group_by="selection", benign_data_type="vanilla_benign",
                                  seed=7, max_length=100)
        with self.assertRaises(ValueError):
            helper.build_training_data(args, FakeTokenizer())

    def test_planner_lists_the_seeds_subsets(self):
        for name in ("lexical_hdd", "semantic_qdit_bge-m3"):
            self.save(name, self.pool.iloc[:6])
        self.save("other_size", self.pool.iloc[:5])
        self.save("other_seed", self.pool.iloc[:6], seed=8)
        self.assertEqual(train_jobs.dataset_groups("aegis", 6, "selection", seed=7),
                         (["lexical_hdd", "semantic_qdit_bge-m3"], {}))
        self.assertEqual(train_jobs.dataset_groups("aegis", 6, "selection",
                                                   ["semantic_qdit_bge-m3", "unknown"], seed=7),
                         (["semantic_qdit_bge-m3"], {}))
        self.assertEqual(train_jobs.dataset_groups("gretel", 6, "selection", seed=7), ([], {}))
        name = helper.checkpoint_name("llama3.1_git20k", 1800, "aegis",
                                      "semantic_vendi_qwen3-embedding-0.6b", 7)
        import eval as evaluation

        safety_dir, if_dir, prefix = evaluation.result_dirs("out", name)
        self.assertEqual(safety_dir, "out/seed7/llama3.1_git20k_1800_aegis/"
                                     "aegis_semantic_vendi_qwen3-embedding-0.6b")
        self.assertEqual(prefix, "aegis_semantic_vendi_qwen3-embedding-0.6b")


if __name__ == "__main__":
    unittest.main()
