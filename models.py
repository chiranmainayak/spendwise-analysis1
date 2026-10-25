from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text
from .database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    month_sheet = Column(String(30), index=True, nullable=False)
    date = Column(String(10), index=True, nullable=False)  # YYYY-MM-DD
    time = Column(String(8), nullable=False)               # HH:MM
    amount_debited = Column(Float, nullable=False)
    category = Column(String(50), index=True, nullable=False)
    amount_credited = Column(Float, default=0.0)
    remaining_balance = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelBenchmark(Base):
    __tablename__ = "model_benchmarks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False)
    r2_score = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    mae = Column(Float, nullable=False)
    is_primary = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    trained_at = Column(DateTime, default=datetime.utcnow)


class ClassifierMetric(Base):
    __tablename__ = "classifier_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), default="CategoryClassifier")
    accuracy = Column(Float, nullable=False)
    notes = Column(Text, nullable=True)
    trained_at = Column(DateTime, default=datetime.utcnow)


class RetrainState(Base):
    __tablename__ = "retrain_states"

    id = Column(Integer, primary_key=True, default=1)
    new_transactions_since_retrain = Column(Integer, default=0)
    total_transactions = Column(Integer, default=0)
    last_retrained_at = Column(DateTime, default=datetime.utcnow)
