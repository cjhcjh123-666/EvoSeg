"""Auditable edit-state primitives; bookkeeping is not a novelty claim.

Gold object identifiers below belong to offline supervision only. A deployed
router must resolve references against its own predicted history, never GT IDs.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SelectionLedger:
    known_targets: frozenset[str]
    active: frozenset[str] = frozenset()
    snapshots: list[frozenset[str]] = field(default_factory=list)

    def apply(self, operation, targets=()):
        selected = frozenset(targets)
        if not selected <= self.known_targets:
            raise ValueError('edit refers to an unknown target')
        if operation == 'undo':
            if selected or not self.snapshots:
                raise ValueError('undo needs a previous edit and no new target')
            self.active = self.snapshots.pop()
            return self.active
        if not selected:
            raise ValueError('an edit needs an explicit resolved target')
        if operation == 'select':
            updated = selected
        elif operation == 'add':
            if selected <= self.active:
                raise ValueError('addition has no effect')
            updated = self.active | selected
        elif operation == 'remove':
            if not selected <= self.active:
                raise ValueError('removal target is not currently selected')
            updated = self.active - selected
        else:
            raise ValueError('unsupported edit operation')
        self.snapshots.append(self.active)
        self.active = updated
        return self.active


def mask_recipe(active, registry):
    """Union existing public annotation tracks; never invent or subtract GT pixels."""
    return sorted({str(anno) for key in active for anno in registry[key]['anno_ids']})


def compose_scoped_mask(previous, proposal, scope):
    """Differentiable primitive; scope==0 preserves previous pixels exactly.

    Learning scope from instructions/history is the pending research component.
    This operation alone is not a claimed paper contribution.
    """
    return previous + scope * (proposal - previous)
