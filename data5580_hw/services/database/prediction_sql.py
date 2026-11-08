from sqlalchemy import UniqueConstraint, Index, CLOB, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from typing import List
from data5580_hw.services.database.database_client import db

class ModelSQL(db.Model):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)
    model_type: Mapped[str] = mapped_column(db.String(120), nullable=True)
    model_name: Mapped[str] = mapped_column(db.String(120), nullable=True)
    model_version: Mapped[str] = mapped_column(db.String(120), nullable=True)

    updated: Mapped[datetime] = mapped_column(db.DateTime)
    created: Mapped[datetime] = mapped_column(db.DateTime)

    predictions: Mapped[List['PredictionSQL']] = db.relationship("PredictionSQL", back_populates="model")

class PredictionSQL(db.Model):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)
    features: Mapped[dict] = mapped_column(CLOB, nullable=False)
    score: Mapped[float] = mapped_column(db.Numeric, nullable=True)
    threshold: Mapped[float] = mapped_column(db.Numeric, nullable=True)
    label_str: Mapped[str] = mapped_column(db.String(120), nullable=True)
    label_numeric: Mapped[float] = mapped_column(db.Numeric, nullable=True)
    actual_str: Mapped[str] = mapped_column(db.String(120), nullable=True)
    actual_numeric: Mapped[float] = mapped_column(db.Numeric, nullable=True)

    updated: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)
    created: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)

    model_id: Mapped[str] = mapped_column(db.String(120), db.ForeignKey("models.id"))
    model: Mapped['ModelSQL'] = db.relationship("ModelSQL", back_populates="predictions")

