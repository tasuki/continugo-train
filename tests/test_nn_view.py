import base64
import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from test_features import SGF

from continugo_train.features import COARSE_PLANES, MOVE_FEATURES, FeatureClient, inspect_game
from continugo_train.model import NetworkConfig, PolicyValueNet, cell_actions
from continugo_train.nn_view import (
    RATED_MODE,
    frame,
    generate,
    generate_rated,
    heatmap,
    json_safe,
    load_checkpoint,
    raster,
)
from continugo_train.train import feature_spec


def report_data(path):
    text = path.read_text()
    return text, json.loads(
        text.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    )


class NnViewerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sgf = self.root / "game.sgf"
        self.sgf.write_text(SGF)
        self.checkpoint = self.root / "checkpoint.pt"
        self.model = PolicyValueNet(
            NetworkConfig(
                channels=4,
                blocks=1,
                policy_hidden=4,
                value_hidden=4,
                score_hidden=4,
                ownership_hidden=4,
                top_cells=2,
            )
        ).eval()
        torch.save(
            {
                "config": asdict(self.model.config),
                "model": self.model.state_dict(),
                "feature_spec": feature_spec(4),
            },
            self.checkpoint,
        )

    def test_self_contained_report_all_moves_inputs_images_and_navigation(self):
        output = self.root / "game.html"
        generate(self.sgf, self.checkpoint, output)
        text, data = report_data(output)
        self.assertEqual(len(data["frames"]), 3)
        self.assertEqual(data["features"], list(MOVE_FEATURES))
        self.assertEqual(data["sgf_sha256"], hashlib.sha256(self.sgf.read_bytes()).hexdigest())
        self.assertEqual(base64.b64decode(data["sgf"].split(",")[1]), self.sgf.read_bytes())
        self.assertIn("ArrowLeft", text)
        self.assertIn("ArrowRight", text)
        self.assertNotIn('src="http', text)
        for ply, f in enumerate(data["frames"]):
            self.assertEqual(f["ply"], ply)
            self.assertEqual(f["player"], ply % 2)
            self.assertEqual([image["name"] for image in f["coarse"]], list(COARSE_PLANES))
            self.assertEqual([image["name"] for image in f["exact"]], list(MOVE_FEATURES))
            self.assertEqual(f["outcome"], 1 if ply % 2 == 0 else -1)
            for group in ("coarse", "exact", "policy", "heads"):
                for image in f[group]:
                    self.assertTrue(
                        base64.b64decode(image["src"].split(",")[1]).startswith(b"\x89PNG")
                    )
        self.assertEqual(data["frames"][1]["view"]["stones"], [[200, 200, 0]])
        self.assertEqual(data["frames"][0]["view"]["action"], 200200)
        self.assertEqual(data["frames"][1]["coarse"][5]["high"], 1)
        self.assertEqual(data["frames"][1]["coarse"][5]["name"], "enemy_liberty_centers")
        self.assertEqual(data["frames"][1]["coarse"][7]["name"], "enemy_liberty_footprints")
        with self.assertRaises(FileExistsError):
            generate(self.sgf, self.checkpoint, output)
        self.assertEqual(output.read_text(), text)

    def test_complete_cell_requests_and_mixed_trailers_recover_after_errors(self):
        with FeatureClient(profile=True) as client:
            sample = client.load(self.sgf, 1, cells=[4, 4, 5], view=True)
            torch.testing.assert_close(sample.actions, cell_actions([4, 5], sample.resolution))
            self.assertEqual(client.last_view["action"], 1000000)
            self.assertIn("server_coarse_seconds", client.last_profile)
            with self.assertRaisesRegex(ValueError, "outside"):
                client.load(self.sgf, 0, cells=[999999], view=True)
            self.assertIsNone(client.last_view)
            client.profile = False
            client.load(self.sgf, 2, cells=[], view=True)
            self.assertIsNotNone(client.last_view)
            client.load(self.sgf, 0)
            self.assertIsNone(client.last_view)

    def test_optional_group_geometry_has_rules_membership_connections_and_liberties(self):
        self.sgf.write_text("(;FF[9071]GM[9071]SZ[13];B[dSdS];W[];B[fqdS];W[];B[])")
        with FeatureClient(profile=True) as client:
            plain = client.load(self.sgf, 3, cells=[], view=True)
            self.assertNotIn("groups", client.last_view)
            detailed = client.load(self.sgf, 3, cells=[], view_geometry=True)
            torch.testing.assert_close(plain.coarse, detailed.coarse, rtol=0, atol=0)
            (group,) = client.last_view["groups"]
            self.assertEqual(group["player"], 0)
            self.assertEqual(set(group["members"]), {0, 1})
            self.assertEqual(group["connections"], [[0, 1]])
            self.assertTrue(group["liberty_spans"])
            self.assertTrue(all(left <= right for _, left, right in group["liberty_spans"]))
            legacy = client.load(self.sgf, 3, cells=[], feature_schema=4)
            self.assertAlmostEqual(group["liberty_area_stones"], float(legacy.coarse[5].max()))
            self.assertAlmostEqual(group["size_feature"], float(legacy.coarse[3].max()))
            self.assertIn("server_coarse_seconds", client.last_profile)
            client.load(self.sgf, 3, cells=[], view=True)
            self.assertNotIn("groups", client.last_view)  # no unread geometry trailer

    def test_frame_critics_match_eager_and_policy_mass_is_preserved(self):
        info = inspect_game(self.sgf)
        loaded = load_checkpoint(self.checkpoint)
        with FeatureClient() as client, torch.inference_mode():
            f = frame(client, self.sgf, 1, loaded, info)
            sample = client.load(self.sgf, 1, cells=[])
            _, _, value, score, _ = self.model.encode_training(sample.coarse[None])
        self.assertAlmostEqual(f["value"], value.item())
        self.assertAlmostEqual(f["score"], score.item())
        self.assertAlmostEqual(f["pass"], f["coarse_pass"] if f["legal"] else 1.0)
        self.assertAlmostEqual(f["policy_mass"], 1.0, places=5)
        self.assertAlmostEqual(f["utility"], (value.item() + score.item()) / 2)
        self.assertLessEqual(sum(row["p"] for row in f["candidates"]), 1.00001)

    def test_truncation_has_no_fabricated_labels_or_ownership_target(self):
        self.sgf.write_text(SGF.replace("RE[B+220.4]", "").split(";W[]")[0] + ")")
        output = self.root / "truncated.html"
        generate(self.sgf, self.checkpoint, output)
        _, data = report_data(output)
        f = data["frames"][0]
        self.assertIsNone(f["outcome"])
        self.assertIsNone(f["score_target"])
        self.assertFalse(any("target" in image["name"] for image in f["heads"]))

    def test_color_specific_checkpoint_identity_and_random_turns_are_skipped(self):
        identity = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.sgf.write_text(
            '(;FF[9071]GM[9071]SZ[13]C[{"model":"'
            + identity
            + '","opponent":"'
            + "0" * 64
            + '"}];B[];W[])'
        )
        with self.assertRaisesRegex(ValueError, "does not match SGF opponent"):
            generate(self.sgf, self.checkpoint, self.root / "bad.html")
        self.assertFalse((self.root / "bad.html").exists())
        self.sgf.write_text(self.sgf.read_text().replace("0" * 64, "random-no-eye-fill-v1"))
        generate(self.sgf, self.checkpoint, self.root / "reference.html")
        _, data = report_data(self.root / "reference.html")
        self.assertFalse(data["frames"][0]["counterfactual"])
        self.assertEqual(len(data["frames"]), 1)
        self.assertEqual(data["skipped"][0]["ply"], 1)
        self.assertIn("Random bot", data["skipped"][0]["reason"])

    def test_distinct_white_checkpoint_is_used_and_terminal_view_framing_recovers(self):
        black_hash = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        with torch.no_grad():
            self.model.value_head[2].bias.fill_(0.8)
        white = self.root / "white.pt"
        torch.save(
            {
                "config": asdict(self.model.config),
                "model": self.model.state_dict(),
                "feature_spec": feature_spec(4),
            },
            white,
        )
        white_hash = hashlib.sha256(white.read_bytes()).hexdigest()
        self.sgf.write_text(
            '(;FF[9071]GM[9071]SZ[13]C[{"model":"'
            + black_hash
            + '","opponent":"'
            + white_hash
            + '"}];B[];W[])'
        )
        generate(self.sgf, self.checkpoint, self.root / "pair.html", white_checkpoint=white)
        _, data = report_data(self.root / "pair.html")
        self.assertEqual([f["model"] for f in data["frames"]], [black_hash, white_hash])
        with FeatureClient() as client, torch.inference_mode():
            sample = client.load(self.sgf, 1, cells=[])
            expected = self.model.encode_search(sample.coarse[None])[2].item()
            self.assertAlmostEqual(data["frames"][1]["value"], expected)
            with self.assertRaisesRegex(ValueError, "terminal"):
                client.load(self.sgf, 2, cells=[], view=True)
            client.load(self.sgf, 1, cells=[], view=True)
            self.assertEqual(client.last_view["action"], 1000000)

    def test_rated_partial_and_empty_reports_and_identity_failures(self):
        identity = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        unsupported = self.root / "unsupported.pt"
        torch.save({"feature_spec": {"samples_per_diameter": 4, "schema": 999}}, unsupported)
        other = hashlib.sha256(unsupported.read_bytes()).hexdigest()
        self.sgf.write_text(
            '(;FF[9071]GM[9071]SZ[13]C[{"model":"'
            + identity
            + '","opponent":"'
            + other
            + '"}];B[];W[];B[])'
        )
        # Use a nonterminal third move (two opening passes would finish the game).
        self.sgf.write_text(self.sgf.read_text().replace(";B[];W[];B[]", ";B[dSdS];W[];B[]"))
        info = inspect_game(self.sgf)
        paths = {identity: self.checkpoint, other: unsupported}
        generate_rated(self.sgf, self.root / "partial.html", info, paths)
        _, data = report_data(self.root / "partial.html")
        self.assertEqual(data["mode"], RATED_MODE)
        self.assertEqual([f["ply"] for f in data["frames"]], [0, 2])
        self.assertTrue(
            all(f["model"] == identity and not f["counterfactual"] for f in data["frames"])
        )
        self.assertEqual([s["ply"] for s in data["skipped"]], [1])
        self.assertIn("Unsupported", data["skipped"][0]["reason"])
        generate_rated(self.sgf, self.root / "empty.html", info, {})
        text, empty = report_data(self.root / "empty.html")
        self.assertEqual(empty["frames"], [])
        self.assertEqual(len(empty["skipped"]), 3)
        self.assertIn("No moves can be visualized", text)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            generate_rated(self.sgf, self.root / "bad-rated.html", info, {identity: unsupported})

    def test_legacy_viewers_keep_all_original_inputs(self):
        old_model = PolicyValueNet(replace(self.model.config, input_planes=12)).eval()
        for schema in (4,):
            saved = {
                "config": asdict(old_model.config),
                "model": old_model.state_dict(),
                "feature_spec": feature_spec(4, schema=schema),
            }
            old = self.root / f"legacy-schema{schema}.pt"
            torch.save(saved, old)
            loaded = load_checkpoint(old)
            self.assertEqual(loaded[0].feature_schema, schema)
            with FeatureClient() as client, torch.inference_mode():
                f = frame(client, self.sgf, 1, loaded, inspect_game(self.sgf))
                sample = client.load(self.sgf, 1, cells=[], feature_schema=schema)
                expected = old_model.encode_search(sample.coarse[None])[2].item()
            self.assertEqual(f["feature_schema"], schema)
            self.assertEqual(len(f["coarse"]), 12)
            self.assertGreater(f["coarse"][5]["high"], 1)  # Legacy scalar is not clipped.
            self.assertEqual(f["coarse"][6]["name"], "friendly_capturable_in_one")
            self.assertEqual(f["coarse"][1]["note"], "Point-sampled")
            self.assertAlmostEqual(f["value"], expected)

    def test_png_rgba_transparency_binning_and_safe_json(self):
        image = heatmap("test", np.array([[0.0, np.nan], [1.0, 0.5]]), limits=(0, 1))
        png = base64.b64decode(image["src"].split(",")[1])
        self.assertEqual(struct.unpack(">2I", png[16:24]), (2, 2))
        # IHDR ends at byte 33, then IDAT length + type + zlib scanlines.
        length = struct.unpack(">I", png[33:37])[0]
        raw = zlib.decompress(png[41 : 41 + length])
        self.assertEqual(raw[8], 0)  # alpha of first row's second pixel
        image = raster([500500, 500501], [0.2, 0.3], mode="sum", width=10)
        self.assertAlmostEqual(np.nansum(image), 0.5)
        self.assertNotIn("</script>", json_safe({"name": "</script>"}))
        with self.assertRaises(ValueError):
            json_safe({"bad": float("nan")})


if __name__ == "__main__":
    unittest.main()
