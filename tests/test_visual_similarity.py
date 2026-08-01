"""Tests for the visual_similarity package: rendering, embeddings, ranking, pair metrics.

Uses the committed font archive (extracted on demand) and a small untrained
small_cnn, so no real checkpoint or full label set is needed.
"""

from __future__ import annotations

import json

import pytest
import torch

from char_recognition.models.recognizer import CharRecognizer
from char_recognition.paths import JP_FONT
from char_recognition.visual_similarity.embed import (
    aggregate_fonts,
    blend_embeddings,
    class_weight_embedding,
    extract_image_embeddings,
    image_embedding_names,
)
from char_recognition.visual_similarity.pair_evaluation import evaluate_pairs, load_lookalike_pairs
from char_recognition.visual_similarity.render import extract_fonts, render_glyphs
from char_recognition.visual_similarity.similarity_search import (
    pair_rank,
    top_k_similar,
    write_neighbours_json,
)

LABELS = list("口日目田由甲申百白千干土士人入大犬太力九石右未末")  # 24 distinct chars


@pytest.fixture(scope="module")
def serif_font():
    if not JP_FONT.exists():
        extract_fonts()
    return JP_FONT


@pytest.fixture(scope="module")
def model() -> CharRecognizer:
    torch.manual_seed(0)
    return CharRecognizer(
        len(LABELS), backbone="small_cnn", in_channels=1, image_size=(64, 64),
        backbone_kwargs={"head_rank": 8},
    ).eval()


class TestRender:
    def test_shape_range_and_polarity(self, serif_font):
        images, covered = render_glyphs(LABELS, serif_font)
        assert images.shape == (len(LABELS), 1, 64, 64)
        assert bool(covered.all())
        # White ink on black background: corners dark, ink bright, range within [0, 255].
        assert float(images.min()) == 0.0 and float(images.max()) == 255.0
        assert float(images[:, :, 0, 0].max()) == 0.0  # corners are background
        assert float(images.amax(dim=(1, 2, 3)).min()) > 0  # every glyph drew something

    def test_uncovered_label_is_masked_and_blank(self, serif_font):
        images, covered = render_glyphs(["日", "\U000e0001"], serif_font)  # tag char: no glyph
        assert covered.tolist() == [True, False]
        assert float(images[1].sum()) == 0.0

    def test_missing_font_points_to_setup(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"setup\.py"):
            render_glyphs(["日"], tmp_path / "nope.ttf")


class TestEmbed:
    def test_embedding_shapes(self, model):
        images = torch.randint(0, 256, (5, 1, 64, 64)).float()
        extracted = extract_image_embeddings(model, images, batch_size=2)
        dims = {"f3": 64, "f5": 128, "f5_4x4": 2048, "f7": 256, "pool": 256, "head": 8}
        assert set(extracted) == set(image_embedding_names(model)) == set(dims)
        for name, dim in dims.items():
            assert extracted[name].shape == (5, dim)

    def test_batching_is_consistent(self, model):
        images = torch.randint(0, 256, (4, 1, 64, 64)).float()
        one = extract_image_embeddings(model, images, batch_size=4)
        two = extract_image_embeddings(model, images, batch_size=2)
        for name in one:
            torch.testing.assert_close(one[name], two[name])

    def test_class_weight_embedding(self, model):
        weights = class_weight_embedding(model)
        assert weights.shape == (len(LABELS), 8)  # head_rank columns
        full = CharRecognizer(7, backbone="small_cnn", in_channels=1, image_size=(64, 64))
        assert class_weight_embedding(full).shape == (7, 256)  # full head: feat columns

    def test_non_small_cnn_rejected(self):
        tiny = CharRecognizer(4, backbone="tiny_cnn", in_channels=1, image_size=(32, 32))
        with pytest.raises(TypeError, match="small_cnn"):
            image_embedding_names(tiny)

    def test_blend_embeddings(self):
        a = torch.tensor([[2.0, 0.0], [0.0, 0.0]])  # row 1 invalid (zero)
        b = torch.tensor([[0.0, 3.0], [5.0, 0.0]])
        blended = blend_embeddings(a, b)
        torch.testing.assert_close(blended[0], torch.tensor([0.7071, 0.7071]), atol=1e-3, rtol=0)
        torch.testing.assert_close(blended[1], torch.tensor([1.0, 0.0]))  # falls back to b

    def test_aggregate_fonts_masked_mean(self):
        a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0], [9.0, 9.0]])  # row 1 of font b is uncovered
        emb, valid = aggregate_fonts([a, b], [torch.tensor([True, True]), torch.tensor([True, False])])
        torch.testing.assert_close(emb[0], torch.tensor([0.7071, 0.7071]), atol=1e-3, rtol=0)
        torch.testing.assert_close(emb[1], torch.tensor([1.0, 0.0]))  # only font a counted
        assert valid.tolist() == [True, True]
        _, none_valid = aggregate_fonts([a], [torch.tensor([False, False])])
        assert none_valid.tolist() == [False, False]


class TestSimilaritySearch:
    EMB = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])

    def test_topk_excludes_self_and_orders(self):
        indices, scores = top_k_similar(self.EMB, k=2)
        assert indices[0].tolist() == [1, 3]  # nearest first, never itself
        assert scores[0][0] > scores[0][1]

    def test_invalid_rows_never_appear(self):
        valid = torch.tensor([True, False, True, True])
        indices, _ = top_k_similar(self.EMB, k=2, valid=valid)
        assert 1 not in indices[[0, 2, 3]].flatten().tolist()

    def test_pair_rank(self):
        assert pair_rank(self.EMB, 0, 1) == 1
        assert pair_rank(self.EMB, 0, 2) == 3
        assert pair_rank(self.EMB, 0, 2, valid=torch.tensor([True, False, True, True])) == 2

    def test_json_artifact(self, tmp_path):
        indices, scores = top_k_similar(self.EMB, k=2, valid=torch.tensor([True, True, True, False]))
        path = write_neighbours_json(
            tmp_path / "n.json", ["a", "b", "c", "d"], indices, scores,
            valid=torch.tensor([True, True, True, False]),
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["a"][0]["char"] == "b" and isinstance(data["a"][0]["score"], float)
        assert data["d"] == []  # invalid query -> empty list


class TestPairEvaluation:
    def test_pair_file_loads_and_categories_present(self):
        pairs = load_lookalike_pairs("configs/eval/lookalike_pairs.toml")
        assert {"left_right_mirage", "atomic_mirage", "classic_confusables"} <= set(pairs)
        assert ("粗", "相") in pairs["left_right_mirage"]

    def test_metrics_on_known_geometry(self):
        emb = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [-1.0, 0.0]])
        labels = ["a", "b", "c", "d"]
        results = evaluate_pairs(emb, labels, {"close": [("a", "b")], "far": [("a", "d")]}, k=1)
        assert results["close"].hits == 2 and results["close"].mrr == 1.0
        assert results["far"].hits == 0 and results["far"].mrr == pytest.approx(1 / 3)
        assert results["overall"].total == 4

    def test_invalid_class_skipped(self):
        emb = torch.eye(3)
        results = evaluate_pairs(
            emb, ["a", "b", "c"], {"x": [("a", "b")]}, k=1, valid=torch.tensor([True, False, True])
        )
        assert results["x"].total == 0 and results["x"].skipped == 2

    def test_unknown_pair_char_raises(self):
        with pytest.raises(ValueError, match="not in labels"):
            evaluate_pairs(torch.eye(2), ["a", "b"], {"x": [("a", "z")]})
