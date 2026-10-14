"""add_spatial_and_statistical_telemetry

Revision ID: 76cb5e944d05
Revises: None
Create Date: 2026-10-09 09:15:02.430038

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '76cb5e944d05'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to store spatial and statistical telemetry."""
    with op.batch_alter_table('entities', schema=None) as batch_op:
        batch_op.add_column(sa.Column('attack_strength', sa.Float(), nullable=True, server_default='0.0'))
        batch_op.add_column(sa.Column('defense_strength', sa.Float(), nullable=True, server_default='0.0'))
        batch_op.add_column(sa.Column('avg_xg_for', sa.Float(), nullable=True, server_default='0.0'))
        batch_op.add_column(sa.Column('avg_xg_against', sa.Float(), nullable=True, server_default='0.0'))
        batch_op.add_column(sa.Column('avg_corners', sa.Float(), nullable=True, server_default='0.0'))
        batch_op.add_column(sa.Column('form_rating', sa.Float(), nullable=True, server_default='0.0'))

    with op.batch_alter_table('matches', schema=None) as batch_op:
        batch_op.add_column(sa.Column('expected_total_goals', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('home_win_prob_baseline', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('away_win_prob_baseline', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('matches', schema=None) as batch_op:
        batch_op.drop_column('away_win_prob_baseline')
        batch_op.drop_column('home_win_prob_baseline')
        batch_op.drop_column('expected_total_goals')

    with op.batch_alter_table('entities', schema=None) as batch_op:
        batch_op.drop_column('form_rating')
        batch_op.drop_column('avg_corners')
        batch_op.drop_column('avg_xg_against')
        batch_op.drop_column('avg_xg_for')
        batch_op.drop_column('defense_strength')
        batch_op.drop_column('attack_strength')
