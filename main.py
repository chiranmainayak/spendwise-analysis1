import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base, SessionLocal
from .models import ModelBenchmark, Transaction
from .ml.trainer import MLTrainer
from .routes import transactions, analytics, predictions, websocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spendwise.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure database schema exists
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized.")

    # Check and initialize ML benchmarks if empty
    db = SessionLocal()
    try:
        tx_count = db.query(Transaction).count()
        if tx_count > 0:
            benchmarks = db.query(ModelBenchmark).count()
            if benchmarks == 0:
                logger.info("Initializing baseline ML benchmarks on held-out dataset...")
                trainer = MLTrainer(db)
                trainer.train_and_evaluate()
                logger.info("ML benchmarks initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not auto-initialize ML benchmarks on startup: {e}")
    finally:
        db.close()

    yield
    logger.info("SpendWise Live backend shutting down.")


app = FastAPI(
    title="SpendWise Live API",
    description="Real-Time Student Expense Tracker with ML Overspend Projections & WebSocket Broadcasts",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(transactions.router)
app.include_router(analytics.router)
app.include_router(predictions.router)
app.include_router(websocket.router)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "app": "SpendWise Live",
        "service": "FastAPI + WebSockets + SQLite",
    }
