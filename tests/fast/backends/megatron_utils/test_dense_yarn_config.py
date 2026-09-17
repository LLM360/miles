from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.yarn import apply_dense_yarn_config


def test_dense_qwen3_yarn_settings_survive_legacy_config_conversion():
    config = SimpleNamespace()
    args = Namespace(
        position_embedding_type="yarn",
        multi_latent_attention=False,
        rotary_scaling_factor=4.0,
        yarn_original_max_position_embeddings=32768,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=0.0,
        yarn_correction_range_round_to_int=True,
    )
    apply_dense_yarn_config(config, args)
    assert vars(config) == {
        "yarn_rotary_scaling_factor": 4.0,
        "yarn_original_max_position_embeddings": 32768,
        "yarn_beta_fast": 32.0,
        "yarn_beta_slow": 1.0,
        "yarn_mscale": 1.0,
        "yarn_mscale_all_dim": 0.0,
        "yarn_correction_range_round_to_int": True,
    }


def test_existing_yarn_config_is_preserved_and_none_uses_defaults():
    config = SimpleNamespace(yarn_rotary_scaling_factor=8.0)
    args = Namespace(position_embedding_type="yarn", rotary_scaling_factor=4.0, mscale=None)
    apply_dense_yarn_config(config, args)
    assert config.yarn_rotary_scaling_factor == 8.0
    assert config.yarn_mscale == 1.0
    assert config.yarn_original_max_position_embeddings == 4096


@pytest.mark.parametrize("position,mla", [("rope", False), ("yarn", True)])
def test_other_rotary_embedding_paths_are_unchanged(position, mla):
    config = SimpleNamespace(existing="keep")
    apply_dense_yarn_config(config, Namespace(position_embedding_type=position, multi_latent_attention=mla))
    assert vars(config) == {"existing": "keep"}
