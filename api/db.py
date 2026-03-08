import os
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Session


# ── Connection ─────────────────────────────────────────────────────────────
#
# create_engine() does NOT open a connection immediately.
# It creates a "connection pool" — a set of reusable connections.
# This matters for a web API: if 50 requests arrive at once, they each
# grab a connection from the pool instead of opening 50 new ones.
#
# The URL comes from .env so we never hardcode credentials in code.
#
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://admin:admin@postgres:5432/fraud_db"  # fallback for local dev
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # test connections before using them (handles Postgres restarts)
    pool_size=5,          # keep 5 connections warm
    max_overflow=10,      # allow up to 10 extra under heavy load
)


# ── Base class ─────────────────────────────────────────────────────────────
#
# All your table classes inherit from Base.
# This is how SQLAlchemy knows "this Python class represents a DB table"
# vs any other class in your codebase.
#
class Base(DeclarativeBase):
    pass


# ── Table: transactions ────────────────────────────────────────────────────
#
# Mirrors the `transactions` table from your init.sql exactly.
# This table stores the raw input data sent to your API.
#
# Notice: __tablename__ must match the SQL table name exactly.
#
class Transaction(Base):
    __tablename__ = "transactions"

    id              = Column(Integer, primary_key=True)
    user_id         = Column(Integer)
    amount          = Column(Float)
    timestamp       = Column(DateTime)
    location        = Column(String(100))
    merchant_type   = Column(String(100))
    is_fraud        = Column(Boolean)


# ── Table: model_predictions ───────────────────────────────────────────────
#
# Mirrors the `model_predictions` table from your init.sql.
# Every time your API makes a prediction, it logs a row here.
# This is your audit trail — you can always go back and see what
# the model decided, when, and with what confidence.
#
class Prediction(Base):
    __tablename__ = "model_predictions"

    id                  = Column(Integer, primary_key=True)
    transaction_id      = Column(Integer)
    fraud_probability   = Column(Float)
    decision            = Column(String(50))
    model_version       = Column(String(50))
    #
    # server_default=func.now() means:
    # "let Postgres fill this in with the current timestamp at insert time"
    # This is better than doing it in Python because the DB timestamp
    # is always consistent, even if your app servers are in different timezones.
    #
    created_at          = Column(DateTime, server_default=func.now())


# ── Session factory ────────────────────────────────────────────────────────
#
# A Session is your "unit of work" with the database.
# It tracks everything you add/change and either commits it all at once
# or rolls it all back if something fails. This is a transaction in the
# database sense.
#
# get_db() is a FastAPI "dependency". Here's the lifecycle:
#
#   1. Request arrives at FastAPI
#   2. FastAPI sees `db = Depends(get_db)` in your route signature
#   3. FastAPI calls get_db(), which opens a Session
#   4. The `yield` hands that session to your route function as `db`
#   5. Your route runs, uses `db` to query/insert
#   6. FastAPI resumes get_db() after yield — the `with` block closes the session
#   7. Connection returns to the pool, ready for the next request
#
# The `with Session(engine) as session:` handles closing automatically,
# even if your route raises an exception.
#
def get_db():
    with Session(engine) as session:
        yield session