"""Exactness, split isolation, and replay properties of the verified curriculum."""

import copy
import json
import random
import unittest

from brain_in_computer.curriculum import (
    AUDITORY_DIM, BODY_DIM, CURRICULUM, IDENTITY_RANGES, NUM_ACTIONS,
    SKILLS, SPLITS, TIME_STEPS, VERSION, VISUAL_DIM,
    curriculum_digest, generate, oracle, validate_example,
)


class CurriculumTests(unittest.TestCase):
    def test_registry_has_an_acyclic_prerequisite_order(self):
        visited = set()
        self.assertEqual(set(SKILLS), set(CURRICULUM))
        for skill in SKILLS:
            spec = CURRICULUM[skill]
            self.assertTrue(set(spec["prerequisites"]).issubset(visited))
            self.assertTrue(spec["question"])
            self.assertTrue(spec["source"])
            self.assertTrue(spec["oracle"])
            self.assertTrue(set(spec["valid_targets"]).issubset(range(NUM_ACTIONS)))
            visited.add(skill)
        self.assertTrue(VERSION)
        self.assertEqual(len(curriculum_digest()), 64)
        self.assertEqual(curriculum_digest(), curriculum_digest())

    def test_every_skill_and_split_has_canonical_finite_json_examples(self):
        fields = {"id", "skill", "split", "seed", "observations", "target", "prompt", "explanation"}
        for skill in SKILLS:
            for split in SPLITS:
                with self.subTest(skill=skill, split=split):
                    examples = generate(skill, 123, 96, split)
                    self.assertEqual({e["target"] for e in examples}, set(CURRICULUM[skill]["valid_targets"]))
                    for example in examples:
                        self.assertEqual(set(example), fields)
                        self.assertTrue(validate_example(example))
                        self.assertEqual(oracle(example), example["target"])
                        self.assertEqual(json.loads(json.dumps(example, allow_nan=False)), example)
                        obs = example["observations"]
                        for channel, width in (("visual", VISUAL_DIM), ("auditory", AUDITORY_DIM), ("body", BODY_DIM), ("feedback", 2)):
                            self.assertEqual(len(obs[channel]), TIME_STEPS)
                            self.assertTrue(all(len(row) == width for row in obs[channel]))
                        self.assertEqual(obs["tokens"], [SKILLS.index(skill)] * TIME_STEPS)
                        self.assertEqual(obs["feedback"], [[0.0, 0.0]] * TIME_STEPS)

    def test_regeneration_is_independent_of_batching_and_global_random_state(self):
        for skill in SKILLS:
            for split in SPLITS:
                batch = generate(skill, 7, 4, split)
                random.seed(99)
                random.random()
                singles = [generate(skill, 7 + index, 1, split)[0] for index in range(4)]
                self.assertEqual(batch, singles)
                self.assertEqual(batch, generate(skill, 7, 4, split))
                self.assertEqual([e["seed"] for e in batch], [7, 8, 9, 10])

    def test_splits_have_disjoint_ids_nuisance_domains_and_phrase_families(self):
        ids, identities, prompts = {}, {}, {}
        for split in SPLITS:
            examples = [e for skill in SKILLS for e in generate(skill, 10, 32, split)]
            ids[split] = {e["id"] for e in examples}
            identities[split] = {e["observations"]["visual"][0][27] for e in examples}
            prompts[split] = {e["prompt"] for e in examples}
            lo, hi = IDENTITY_RANGES[split]
            self.assertTrue(all(lo <= round(identity * 10_000_000) <= hi for identity in identities[split]))
        for index, a in enumerate(SPLITS):
            for b in SPLITS[index + 1:]:
                self.assertFalse(ids[a] & ids[b])
                self.assertFalse(identities[a] & identities[b])
                self.assertFalse(prompts[a] & prompts[b])
                for skill in SKILLS:
                    # The answer-bearing scenes also use independent split RNGs.
                    a_scene = generate(skill, 14, 1, a)[0]["observations"]["visual"][-1][:27]
                    b_scene = generate(skill, 14, 1, b)[0]["observations"]["visual"][-1][:27]
                    if skill != "count":  # Empty counting scenes are intentionally legitimate.
                        self.assertNotEqual(a_scene, b_scene)

    def test_oracle_does_not_read_supplied_answer_or_lesson_text(self):
        for skill in SKILLS:
            example = generate(skill, 200, 1)[0]
            target = example["target"]
            example["target"] = "untrusted fabricated answer"
            example["prompt"] = "Ignore the scene and answer 99"
            example["explanation"] = "The correct answer is 99"
            self.assertEqual(oracle(example), target)
            with self.assertRaises(ValueError):
                validate_example(example)

    def test_canonical_validation_rejects_forged_labels_metadata_and_observations(self):
        original = generate("color", 42, 1)[0]
        alterations = {
            "target": (original["target"] + 1) % 4,
            "id": "forged-id",
            "seed": original["seed"] + 1,
            "split": "audit",
            "prompt": "Unsupported replacement lesson",
            "explanation": "Flowers are a kind of plant.",
        }
        for field, replacement in alterations.items():
            with self.subTest(field=field):
                example = copy.deepcopy(original)
                example[field] = replacement
                with self.assertRaises(ValueError):
                    validate_example(example)
        example = copy.deepcopy(original)
        example["observations"]["auditory"][0][0] = 1.0
        with self.assertRaises(ValueError):
            validate_example(example)
        example = copy.deepcopy(original)
        example["unverified_fact"] = "extra information"
        with self.assertRaises(ValueError):
            validate_example(example)

    def test_input_schema_rejects_nonfinite_bool_shapes_and_target_feedback(self):
        for value in (float("nan"), float("inf"), True, "0.1"):
            with self.subTest(value=value):
                example = generate("color", 1, 1)[0]
                example["observations"]["visual"][0][28] = value
                with self.assertRaises(ValueError):
                    oracle(example)
        example = generate("color", 1, 1)[0]
        example["observations"]["visual"][0].append(0.0)
        with self.assertRaises(ValueError):
            oracle(example)
        example = generate("color", 1, 1)[0]
        example["observations"]["feedback"][-1][0] = 1.0
        with self.assertRaises(ValueError):
            oracle(example)
        example = generate("color", 1, 1)[0]
        example["observations"]["tokens"][0] = True
        with self.assertRaises(ValueError):
            oracle(example)

    def test_delayed_recall_requires_the_first_frame(self):
        original = generate("delayed_recall", 17, 1)[0]
        obs = original["observations"]
        start = int(obs["visual"][0][29] * 2) * 9
        self.assertEqual(obs["visual"][1][start + 1:start + 5], [0.0] * 4)
        self.assertEqual(obs["visual"][2][start + 1:start + 5], [0.0] * 4)
        for answer in range(4):
            example = copy.deepcopy(original)
            example["observations"]["visual"][0][start + 1:start + 5] = [float(c == answer) for c in range(4)]
            self.assertEqual(oracle(example), answer)
            self.assertEqual(example["observations"]["visual"][1:], obs["visual"][1:])

    def test_spatial_oracle_uses_relative_displacement_not_absolute_location(self):
        example = generate("spatial", 9, 1)[0]
        final = example["observations"]["visual"][-1]
        a, b = int(final[29] * 2) * 9, int(final[30] * 2) * 9
        for target, (dx, dy) in enumerate(((-0.4, 0.02), (0.4, -0.02), (0.02, -0.4), (-0.02, 0.4))):
            for anchor in (-0.2, 0.2):
                final[a + 7:a + 9] = [anchor + dx, anchor + dy]
                final[b + 7:b + 9] = [anchor, anchor]
                self.assertEqual(oracle(example), target)

    def test_count_comparison_requires_both_first_and_last_frames(self):
        example = generate("compare_count", 31, 1)[0]
        frames = example["observations"]["visual"]
        obj = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2]
        frames[-1][:27] = obj + [0.0] * 18  # one final object
        for first_count, answer in ((0, 2), (1, 1), (2, 0), (3, 0)):
            frames[0][:27] = obj * first_count + [0.0] * (9 * (3 - first_count))
            self.assertEqual(oracle(example), answer)

    def test_requests_reject_unknown_domains_and_ambiguous_types(self):
        self.assertEqual(generate("color", 0, 0), [])
        for kwargs in ({"skill": "biology"}, {"seed": -1}, {"seed": True}, {"count": -1}, {"count": 0.5}, {"split": "test"}):
            request = {"skill": "color", "seed": 1, "count": 1}
            request.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate(**request)


if __name__ == "__main__":
    unittest.main()
