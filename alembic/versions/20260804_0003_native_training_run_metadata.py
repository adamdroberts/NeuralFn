"""Persist native training compatibility and artifact metadata.

Revision ID: 20260804_0003
Revises: 20260404_0002
Create Date: 2026-08-04
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260804_0003"
down_revision = "20260404_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "training_runs",
        sa.Column("runtime", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "training_runs",
        sa.Column("compatibility_report", sa.JSON(), nullable=True),
    )
    op.add_column(
        "training_runs",
        sa.Column("artifact_metadata", sa.JSON(), nullable=True),
    )
    op.add_column(
        "training_runs",
        sa.Column("checkpoint_path", sa.Text(), nullable=True),
    )

    # Avoid JSON server defaults: they are not portable across the supported
    # SQLite/MySQL dialects. Backfill existing rows through SQLAlchemy's JSON
    # binder, then make the application-owned metadata columns non-null.
    op.execute(sa.text("UPDATE training_runs SET runtime = 'scalar' WHERE runtime IS NULL"))
    op.execute(
        sa.text(
            "UPDATE training_runs SET compatibility_report = '{}' "
            "WHERE compatibility_report IS NULL"
        )
    )
    op.execute(
        sa.text(
            "UPDATE training_runs SET artifact_metadata = '{}' "
            "WHERE artifact_metadata IS NULL"
        )
    )

    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.alter_column(
            "runtime",
            existing_type=sa.String(length=64),
            nullable=False,
        )
        batch_op.alter_column(
            "compatibility_report",
            existing_type=sa.JSON(),
            nullable=False,
        )
        batch_op.alter_column(
            "artifact_metadata",
            existing_type=sa.JSON(),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.drop_column("checkpoint_path")
        batch_op.drop_column("artifact_metadata")
        batch_op.drop_column("compatibility_report")
        batch_op.drop_column("runtime")
