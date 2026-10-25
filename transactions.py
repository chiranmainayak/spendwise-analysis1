import asyncio
import io
import csv
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..config import CATEGORIES, MONTHLY_ALLOWANCE, RETRAIN_INTERVAL, HELD_OUT_SHEET
from ..database import get_db
from ..models import Transaction, RetrainState
from ..schemas import TransactionCreate, TransactionResponse
from ..websocket_manager import ws_manager
from ..ml.predictor import MLPredictor
from ..ml.trainer import MLTrainer

router = APIRouter(prefix="/api/transactions", tags=["transactions"])


def derive_month_sheet(date_str: str) -> str:
    """Derives sheet name like 'Apr_2029' from 'YYYY-MM-DD'."""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.strftime("%b_%Y")
    except Exception:
        return HELD_OUT_SHEET


async def background_retrain_task(db: Session):
    """Background worker to retrain models and broadcast new metrics via WebSocket."""
    try:
        trainer = MLTrainer(db)
        results = trainer.train_and_evaluate()
        await ws_manager.broadcast_models_retrained(results)
    except Exception as e:
        print(f"Background retraining error: {e}")


@router.get("", response_model=List[TransactionResponse])
def get_transactions(
    month_sheet: Optional[str] = Query(None, description="Filter by month sheet e.g. Apr_2029"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    query = db.query(Transaction)
    if month_sheet:
        query = query.filter(Transaction.month_sheet == month_sheet)
    if category:
        query = query.filter(Transaction.category == category)

    return query.order_by(desc(Transaction.date), desc(Transaction.time), desc(Transaction.id)).offset(offset).limit(limit).all()


@router.post("", response_model=TransactionResponse)
async def create_transaction(
    payload: TransactionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    now_dt = datetime.now()
    tx_date = payload.date.strip() if payload.date else now_dt.strftime("%Y-%m-%d")
    tx_time = payload.time.strip() if payload.time else now_dt.strftime("%H:%M")
    tx_month = payload.month_sheet.strip() if payload.month_sheet else derive_month_sheet(tx_date)

    # Calculate remaining balance for this month
    past_month_txs = db.query(Transaction).filter(Transaction.month_sheet == tx_month).all()
    spent_before = sum(t.amount_debited for t in past_month_txs)
    total_spent_now = spent_before + payload.amount
    remaining_balance = max(0.0, MONTHLY_ALLOWANCE - total_spent_now)

    new_tx = Transaction(
        month_sheet=tx_month,
        date=tx_date,
        time=tx_time,
        amount_debited=payload.amount,
        category=payload.category,
        amount_credited=0.0,
        remaining_balance=round(remaining_balance, 2),
        created_at=datetime.now(timezone.utc)
    )
    db.add(new_tx)
    db.commit()
    db.refresh(new_tx)

    # Update retraining state
    retrain_state = db.query(RetrainState).first()
    if not retrain_state:
        retrain_state = RetrainState(id=1, new_transactions_since_retrain=1, total_transactions=1)
        db.add(retrain_state)
    else:
        retrain_state.new_transactions_since_retrain += 1
        retrain_state.total_transactions += 1

    trigger_retrain = retrain_state.new_transactions_since_retrain >= RETRAIN_INTERVAL
    if trigger_retrain:
        retrain_state.new_transactions_since_retrain = 0
        retrain_state.last_retrained_at = datetime.now(timezone.utc)
    db.commit()

    # Calculate predictor metrics & diff
    predictor = MLPredictor(db)
    diff_info = predictor.compute_difference(payload.category, payload.amount)
    overspend_info = predictor.calculate_overspend_projection(tx_month)

    # Real-time WebSocket Broadcast
    tx_dict = {
        "id": new_tx.id,
        "month_sheet": new_tx.month_sheet,
        "date": new_tx.date,
        "time": new_tx.time,
        "amount_debited": new_tx.amount_debited,
        "category": new_tx.category,
        "amount_credited": new_tx.amount_credited,
        "remaining_balance": new_tx.remaining_balance,
        "created_at": new_tx.created_at.isoformat() if new_tx.created_at else None,
    }

    # Broadcast transaction added
    await ws_manager.broadcast_transaction_added(
        transaction=tx_dict,
        month_summary=overspend_info,
        diff_info=diff_info
    )

    # Broadcast overspend alert specifically if triggered
    if overspend_info["is_overspend"]:
        await ws_manager.broadcast_overspend_alert(overspend_info)

    # Queue retrain if threshold reached
    if trigger_retrain:
        background_tasks.add_task(background_retrain_task, db)

    return new_tx


@router.get("/export")
def export_transactions_csv(
    month_sheet: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(Transaction)
    if month_sheet:
        query = query.filter(Transaction.month_sheet == month_sheet)
    if category:
        query = query.filter(Transaction.category == category)

    transactions = query.order_by(Transaction.date.asc(), Transaction.time.asc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Transaction ID",
        "Month Sheet",
        "Date",
        "Time",
        "Category",
        "Amount Debited (₹)",
        "Amount Credited (₹)",
        "Remaining Balance (₹)"
    ])

    for tx in transactions:
        writer.writerow([
            tx.id,
            tx.month_sheet,
            tx.date,
            tx.time,
            tx.category,
            f"{tx.amount_debited:.2f}",
            f"{tx.amount_credited:.2f}",
            f"{tx.remaining_balance:.2f}"
        ])

    output.seek(0)
    filename = f"spendwise_transactions_{month_sheet or 'all'}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
