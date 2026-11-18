"""The review policy that a reproducible record must hold (issue #4750).

``reproduce`` re-reviews a record under the policy stored in its verdict. So a record that
stored easier thresholds than its protocol's would pass its own re-review. A record of the
current protocol must hold the protocol's default policy.
"""

from __future__ import annotations

from openinterpretability.core.schemas import PROTOCOL_VERSION, ReviewPolicy

# The only review policy that a reproducible record may hold. None turns the check off:
# the tests' toy records are reviewed under smaller policies (tests/conftest.py).
REVIEW_POLICY: ReviewPolicy | None = ReviewPolicy()


def policy_problem(policy: ReviewPolicy | None) -> str | None:
    """Why ``policy`` can't be reproduced, or ``None`` if it is the protocol's own."""
    if REVIEW_POLICY is not None and policy != REVIEW_POLICY:
        return f"its review policy is not protocol {PROTOCOL_VERSION}'s default policy"
    return None
