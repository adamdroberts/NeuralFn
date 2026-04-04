"""Add dataset catalog and project grants."""

from alembic import op
import sqlalchemy as sa


revision = "20260404_0002"
down_revision = "20260404_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_assets",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("hf_path", sa.String(length=512), nullable=True),
        sa.Column("hf_split", sa.String(length=255), nullable=True),
        sa.Column("text_column", sa.String(length=255), nullable=False),
        sa.Column("num_tokens", sa.Integer(), nullable=True),
        sa.Column("num_rows", sa.Integer(), nullable=True),
        sa.Column("variant", sa.String(length=255), nullable=True),
        sa.Column("train_shards", sa.Integer(), nullable=True),
        sa.Column("val_shards", sa.Integer(), nullable=True),
        sa.Column("data_format", sa.String(length=64), nullable=True),
        sa.Column("repo_id", sa.String(length=512), nullable=True),
        sa.Column("remote_root_prefix", sa.String(length=255), nullable=True),
        sa.Column("created_by_user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_dataset_assets_name", "dataset_assets", ["name"], unique=True)

    op.create_table(
        "project_dataset_grants",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("project_id", sa.String(length=36), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), sa.ForeignKey("dataset_assets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "dataset_id", name="uq_project_dataset_grant"),
    )
    op.create_index("ix_project_dataset_grants_project_id", "project_dataset_grants", ["project_id"], unique=False)
    op.create_index("ix_project_dataset_grants_dataset_id", "project_dataset_grants", ["dataset_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_project_dataset_grants_dataset_id", table_name="project_dataset_grants")
    op.drop_index("ix_project_dataset_grants_project_id", table_name="project_dataset_grants")
    op.drop_table("project_dataset_grants")

    op.drop_index("ix_dataset_assets_name", table_name="dataset_assets")
    op.drop_table("dataset_assets")
