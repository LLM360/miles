from argparse import Namespace
from miles.utils import wandb_utils


def test_explicit_wandb_entity_takes_precedence():
    args = Namespace(wandb_entity="explicit-entity", wandb_team="legacy-team")

    assert wandb_utils._get_wandb_entity(args) == "explicit-entity"
