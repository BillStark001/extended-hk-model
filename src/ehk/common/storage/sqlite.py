from typing import Type, Literal, Dict, Any, Optional, Union, List
from numpy.typing import DTypeLike

import numpy as np

from sqlalchemy import create_engine, inspect, text, MetaData, LargeBinary
from sqlalchemy.orm import DeclarativeMeta, sessionmaker, DeclarativeBase
from sqlalchemy.types import TypeDecorator
import io
import warnings


class NumpyArrayType(TypeDecorator):
  """SQLAlchemy type for storing numpy arrays as blobs."""
  impl = LargeBinary

  cache_ok = True

  def __init__(self, dtype: Optional[Union[str, DTypeLike]] = None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.force_dtype = np.dtype(dtype) if dtype is not None else None

  def process_bind_param(self, value, dialect):
    if value is None:
      return None
    arr = self._to_numpy(value)
    with io.BytesIO() as buf:
      np.save(buf, arr, allow_pickle=False)
      return buf.getvalue()

  def process_result_value(self, value, dialect):
    if value is None:
      return None
    with io.BytesIO(value) as buf:
      arr: np.ndarray = np.load(buf, allow_pickle=False)
    if self.force_dtype is not None:
      arr = arr.astype(self.force_dtype)
    return arr

  def _to_numpy(self, value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
      arr = value
    elif isinstance(value, list):
      arr = np.array(value)
    else:
      raise ValueError(f"Unsupported type for NumpyArrayType: {type(value)}")
    if self.force_dtype is not None:
      arr = arr.astype(self.force_dtype)
    return arr


def create_db_engine_and_session(db_path: str, base: DeclarativeBase | List[DeclarativeBase] | None = None):
  engine = create_engine(f'sqlite:///{db_path}')
  if isinstance(base, list):
    for b in base:
      b.metadata.create_all(engine)
  elif base is not None:
    base.metadata.create_all(engine)
  Session = sessionmaker(bind=engine)
  return engine, Session()


def create_db_session(db_path: str, base: DeclarativeBase | List[DeclarativeBase] | None = None):
  _, session = create_db_engine_and_session(db_path, base)
  return session


def sync_sqlite_table(
    engine,
    model: Type[DeclarativeMeta],
    extra_columns: Literal['ignore', 'error', 'delete'] = 'ignore',
    notnull_sync: Literal['ignore', 'error', 'warn', 'fix'] = 'fix',
) -> None:
  """
  Synchronize a SQLAlchemy ORM model with a SQLite table schema.
  """
  table_name: str = model.__tablename__  # type: ignore
  metadata = MetaData()
  metadata.reflect(bind=engine, only=[table_name], resolve_fks=False)

  model_table = model.__table__  # type: ignore
  inspector = inspect(engine)

  # Read model columns.
  model_fields: Dict[str, Any] = {col.name: col for col in model_table.columns}
  model_field_names = set(model_fields.keys())

  # Read database columns.
  if table_name not in metadata.tables:
    # Create a missing table directly from the model.
    model_table.create(bind=engine)
    return

  db_columns_info = inspector.get_columns(table_name)
  db_columns = {col['name']: col for col in db_columns_info}
  db_field_names = set(db_columns.keys())

  # Add missing columns.
  for field_name in model_field_names - db_field_names:
    col = model_fields[field_name]
    ddl = f'ALTER TABLE "{table_name}" ADD COLUMN {col.compile(dialect=engine.dialect)}'
    with engine.begin() as conn:
      conn.execute(text(ddl))

  # Report columns absent from the model.
  extra_db_fields = db_field_names - model_field_names
  if extra_db_fields:
    msg = f"Extra columns in the table: {extra_db_fields}"
    if extra_columns == 'error':
      raise ValueError(msg)
    elif extra_columns == 'delete':
      # Older SQLite versions require a table rebuild to drop columns.
      raise NotImplementedError(
          "SQLite does not support DROP COLUMN directly. "
          "Manual migration (create new table, copy data) required."
      )

  # Check NOT NULL constraints.
  mismatched_notnull = []
  for field_name in model_field_names & db_field_names:
    model_col = model_fields[field_name]
    db_col = db_columns[field_name]
    model_notnull = not model_col.nullable
    db_notnull = db_col['nullable'] is False
    if model_notnull != db_notnull:
      mismatched_notnull.append((field_name, model_notnull, db_notnull))

  if mismatched_notnull:
    msg = (
        "Fields with mismatched NOT NULL constraint: " +
        ", ".join([
            f"{field} (model: {model_nn}, db: {db_nn})"
            for field, model_nn, db_nn in mismatched_notnull
        ])
    )
    if notnull_sync == 'error':
      raise ValueError(msg)
    elif notnull_sync == 'warn':
      warnings.warn(msg)
    elif notnull_sync == 'fix':
      for field_name, model_notnull, db_notnull in mismatched_notnull:
        if model_notnull and not db_notnull:
          # This helper only adds constraints after checking existing rows.
          try:
            # Reject the migration if existing rows contain null values.
            with engine.connect() as conn:
              res = conn.execute(text(
                  f'SELECT COUNT(*) FROM "{table_name}" WHERE "{field_name}" IS NULL'
              ))
              null_count = res.scalar()
            if null_count > 0:
              raise RuntimeError(
                  f"Cannot set NOT NULL for column '{field_name}' because it has NULL values."
              )
            # Existing SQLite columns require an explicit table migration.
            raise NotImplementedError(
                f"SQLite does not support ALTER COLUMN to add NOT NULL directly. "
                f"Manual migration required for column '{field_name}'."
            )
          except Exception as e:
            raise RuntimeError(
                f"Failed to sync NOT NULL for column '{field_name}': {e}"
            )
        else:
          # Removing NOT NULL also requires an explicit migration.
          raise NotImplementedError(
              f"Removing NOT NULL from column '{field_name}' on SQLite requires manual migration."
          )

  return
