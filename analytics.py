from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..config import CATEGORIES, MONTHLY_ALLOWANCE, HELD_OUT_SHEET
from ..database import get_db
from ..models import Transaction
from ..schemas import MonthSummaryResponse, CategorySpendingItem, DailyTrendItem
from ..ml.predictor import MLPredictor

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/months", response_model=List[str])
def get_available_months(db: Session = Depends(get_db)):
    """Returns all distinct months in chronological order."""
    results = db.query(Transaction.month_sheet).distinct().all()
    # Sort months by examining minimum date per month
    month_dates = (
        db.query(Transaction.month_sheet, func.min(Transaction.date).label("min_date"))
        .group_by(Transaction.month_sheet)
        .order_by("min_date")
        .all()
    )
    return [m[0] for m in month_dates] if month_dates else [HELD_OUT_SHEET]


@router.get("/summary", response_model=MonthSummaryResponse)
def get_month_summary(
    month_sheet: Optional[str] = Query(None, description="Month sheet name (defaults to latest recorded month)"),
    db: Session = Depends(get_db)
):
    if not month_sheet:
        # Default to latest month sheet
        latest_tx = db.query(Transaction).order_by(Transaction.date.desc()).first()
        month_sheet = latest_tx.month_sheet if latest_tx else HELD_OUT_SHEET

    txs = db.query(Transaction).filter(Transaction.month_sheet == month_sheet).order_by(Transaction.date.asc(), Transaction.time.asc()).all()

    predictor = MLPredictor(db)
    projection = predictor.calculate_overspend_projection(month_sheet)

    # Category spending breakdown
    total_spent = sum(t.amount_debited for t in txs)
    cat_amounts = {cat: 0.0 for cat in CATEGORIES}
    cat_counts = {cat: 0 for cat in CATEGORIES}

    for t in txs:
        cat_amounts[t.category] = cat_amounts.get(t.category, 0.0) + t.amount_debited
        cat_counts[t.category] = cat_counts.get(t.category, 0) + 1

    category_spending = []
    for cat in CATEGORIES:
        amt = cat_amounts[cat]
        cnt = cat_counts[cat]
        pct = (amt / total_spent * 100) if total_spent > 0 else 0.0
        exp_avg = predictor.get_expected_amount(cat)["expected_amount"]
        category_spending.append(
            CategorySpendingItem(
                category=cat,
                amount=round(amt, 2),
                count=cnt,
                percentage=round(pct, 1),
                expected_average=exp_avg
            )
        )

    # Daily cumulative trend
    daily_groups = {}
    for t in txs:
        d_day = int(t.date.split("-")[2]) if "-" in t.date else 1
        daily_groups[d_day] = daily_groups.get(d_day, 0.0) + t.amount_debited

    cumulative = 0.0
    daily_trend = []
    # Build complete timeline up to total days or latest day
    highest_day = max(daily_groups.keys()) if daily_groups else 1
    for day in range(1, highest_day + 1):
        daily_amt = daily_groups.get(day, 0.0)
        cumulative += daily_amt
        # Construct sample date string
        month_prefix = txs[0].date[:7] if txs else "2029-04"
        d_str = f"{month_prefix}-{day:02d}"
        daily_trend.append(
            DailyTrendItem(
                date=d_str,
                day=day,
                daily_amount=round(daily_amt, 2),
                cumulative_amount=round(cumulative, 2)
            )
        )

    return MonthSummaryResponse(
        month_sheet=month_sheet,
        allowance=projection["allowance"],
        spent_so_far=projection["spent_so_far"],
        balance_remaining=projection["balance_remaining"],
        days_elapsed=projection["days_elapsed"],
        remaining_days=projection["remaining_days"],
        total_days=projection["total_days"],
        daily_burn_rate=projection["daily_burn_rate"],
        predicted_remaining=projection["predicted_remaining"],
        projected_total=projection["projected_total"],
        is_overspend=projection["is_overspend"],
        projected_overspend_amount=projection["projected_overspend_amount"],
        overspend_percentage=projection["overspend_percentage"],
        alert_message=projection["alert_message"],
        category_spending=category_spending,
        daily_trend=daily_trend
    )
