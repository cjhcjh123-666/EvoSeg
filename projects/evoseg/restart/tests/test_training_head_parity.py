import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from projects.evoseg.restart.native_model import native_training_masks
from projects.evoseg.restart.tests.test_native_forward import TinySAM


def upstream_training_function():
    # Execute only the vendored author's standalone method, not legacy model
    # constructors or FTG dependencies. Its body is the parity reference.
    source = Path(__file__).resolve().parents[3] / 'sa2va/models/extension/sam2_base.py'
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SAM2Base')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_forward_sam_heads')
    namespace = {'torch': torch, 'F': F, 'NO_OBJ_SCORE': -1024.}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_forward_sam_heads']


@pytest.mark.parametrize('multimask', [False, True])
def test_training_head_matches_author_even_with_negative_presence(multimask):
    sam = TinySAM()
    image = torch.ones(2, 1, 1, 1)
    high_res = [torch.ones(2, 1, 2, 2)]
    language = torch.tensor([[[.2]], [[.3]]], requires_grad=True)
    expected = upstream_training_function()(sam, image, high_res_features=high_res,
                                             language_embd=language, multimask_output=multimask)[3]
    actual = native_training_masks(sam, image, high_res, language, multimask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    F.binary_cross_entropy_with_logits(actual, torch.ones_like(actual)).backward()
    assert language.grad.isfinite().all() and language.grad.abs().sum() > 0
    assert (actual > -1024).all()
