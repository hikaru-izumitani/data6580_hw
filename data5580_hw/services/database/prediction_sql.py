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

    predictions: Mapped[List['PredictionSQL']] = relationship("PredictionSQL", back_populates="model")



class ShapSQL(db.Model):
    __tablename__ = 'shaps'

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)
    prediction_id: Mapped[str] = mapped_column(db.String(120), ForeignKey('predictions.id'))
    label: Mapped[str] = mapped_column(db.String(120), nullable=True)
    shap_values: Mapped[dict] = mapped_column(db.CLOB)
    updated: Mapped[datetime] = mapped_column(db.DateTime, default=datetime.now())
    created: Mapped[datetime] = mapped_column(db.DateTime, default=datetime.now())
    prediction: Mapped['PredictionSQL'] = db.relationship("PredictionSQL",back_populates="shaps")




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

    updated: Mapped[datetime] = mapped_column(db.DateTime)
    created: Mapped[datetime] = mapped_column(db.DateTime)

    model_id: Mapped['ModelSQL'] = mapped_column(db.String(120), ForeignKey("models.id"))
    model: Mapped['ModelSQL'] = db.relationship(back_populates="predictions")
    shaps: Mapped[List['ShapSQL']] = relationship("ShapSQL", back_populates="prediction", cascade="all, delete-orphan")