"""Initial migration

Revision ID: c700a293d9a0
Revises: 
Create Date: 2024-11-25 14:04:46.034765

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c700a293d9a0'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('code_files',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('filename', sa.String(), nullable=True),
    sa.Column('content', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_code_files_filename'), 'code_files', ['filename'], unique=False)
    op.create_index(op.f('ix_code_files_id'), 'code_files', ['id'], unique=False)
    op.create_table('vector_embeddings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('code_file_id', sa.Integer(), nullable=False),
    sa.Column('dimension_index', sa.Integer(), nullable=False),
    sa.Column('value', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['code_file_id'], ['code_files.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_vector_embeddings_id'), 'vector_embeddings', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_vector_embeddings_id'), table_name='vector_embeddings')
    op.drop_table('vector_embeddings')
    op.drop_index(op.f('ix_code_files_id'), table_name='code_files')
    op.drop_index(op.f('ix_code_files_filename'), table_name='code_files')
    op.drop_table('code_files')
    # ### end Alembic commands ###