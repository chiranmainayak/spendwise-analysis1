import calendar
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy.orm import Session

from ..config import CATEGORIES, MONTHLY_ALLOWANCE, HELD_OUT_SHEET
from ..models import Transaction, ModelBenchmark, ClassifierMetric


class MLPredictor:
    def __init__(self, db: Session):
        self.db = db
        self._category_means: Dict[str, float] = {}
        self._classifier: Optional[RandomForestClassifier] = None
        self._classifier_classes: List[str] = []
        self._classifier_accuracy: float = 0.50
        self._refresh_models()

    def _refresh_models(self):
        """Precomputes category means and fits fast in-memory classifier."""
        # 1. Fetch training transactions (exclude held-out sheet)
        records = self.db.query(Transaction).filter(Transaction.month_sheet != HELD_OUT_SHEET).all()
        if not records:
            # Fallback to all transactions if held-out is the only one
            records = self.db.query(Transaction).all()

        if not records:
            # Defaults if empty DB
            self._category_means = {cat: 100.0 for cat in CATEGORIES}
            return

        data = [{"amount_debited": r.amount_debited, "category": r.category} for r in records]
        df = pd.DataFrame(data)

        # Precompute means
        means = df.groupby("category")["amount_debited"].mean().to_dict()
        overall_mean = float(df["amount_debited"].mean())
        for cat in CATEGORIES:
            self._category_means[cat] = float(means.get(cat, overall_mean))

        # Train fast classifier for real-time category suggestion
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(df[["amount_debited"]], df["category"])
        self._classifier = clf
        self._classifier_classes = list(clf.classes_)

        # Fetch stored accuracy from DB
        metric = self.db.query(ClassifierMetric).order_by(ClassifierMetric.id.desc()).first()
        if metric:
            self._classifier_accuracy = metric.accuracy

    def get_expected_amount(self, category: str) -> Dict[str, Any]:
        """Returns the training set expected average for a selected category."""
        clean_cat = category.strip()
        expected = self._category_means.get(clean_cat, 100.0)
        return {
            "category": clean_cat,
            "expected_amount": round(expected, 2),
            "formatted_expected": f"₹{expected:,.2f}",
        }

    def compute_difference(self, category: str, actual_amount: float) -> Dict[str, Any]:
        """Calculates difference between actual amount and expected historical category mean."""
        expected = self._category_means.get(category.strip(), 100.0)
        diff = round(actual_amount - expected, 2)
        diff_abs = abs(diff)

        if diff > 0:
            status = "above"
            text = f"₹{diff_abs:,.2f} above expected (₹{expected:,.2f})"
        elif diff < 0:
            status = "below"
            text = f"₹{diff_abs:,.2f} below expected (₹{expected:,.2f})"
        else:
            status = "exact"
            text = f"Exact match with expected (₹{expected:,.2f})"

        return {
            "category": category,
            "actual_amount": actual_amount,
            "expected_amount": round(expected, 2),
            "difference": diff,
            "status": status,
            "summary_text": text,
        }

    def suggest_category(self, amount: float) -> Dict[str, Any]:
        """Given an amount, predicts the most probable category with probabilities."""
        if not self._classifier:
            self._refresh_models()

        if amount <= 0 or not self._classifier:
            return {
                "suggested_category": "Food",
                "probability": 0.0,
                "confidence_percent": "0%",
                "candidates": [],
                "model_accuracy": round(self._classifier_accuracy * 100, 1),
                "disclaimer": "Category suggestion is probabilistic based on past spending patterns."
            }

        probs = self._classifier.predict_proba([[amount]])[0]
        classes = self._classifier_classes

        # Build sorted list of candidates
        candidates = []
        for cat, prob in zip(classes, probs):
            candidates.append({
                "category": cat,
                "probability": round(float(prob), 4),
                "confidence_percent": f"{round(float(prob) * 100, 1)}%"
            })
        candidates.sort(key=lambda x: x["probability"], reverse=True)

        top = candidates[0]
        return {
            "suggested_category": top["category"],
            "probability": top["probability"],
            "confidence_percent": top["confidence_percent"],
            "candidates": candidates,
            "model_accuracy": round(self._classifier_accuracy * 100, 1),
            "disclaimer": "Category suggestion is probabilistic based on past spending patterns."
        }

    def calculate_overspend_projection(self, month_sheet: str) -> Dict[str, Any]:
        """Calculates month projection: spent so far + predicted remainder vs ₹5,000 allowance."""
        txs = self.db.query(Transaction).filter(Transaction.month_sheet == month_sheet).order_by(Transaction.date.asc()).all()
        spent_so_far = sum(t.amount_debited for t in txs)
        balance_remaining = max(0.0, MONTHLY_ALLOWANCE - spent_so_far)

        # Parse days in month
        try:
            # month_sheet format e.g. "Apr_2029"
            parts = month_sheet.split("_")
            month_abbr, year_str = parts[0], parts[1]
            month_num = datetime.strptime(month_abbr, "%b").month
            year_num = int(year_str)
            _, total_days_in_month = calendar.monthrange(year_num, month_num)
        except Exception:
            total_days_in_month = 30

        # Determine days elapsed
        if txs:
            # Extract highest day from transaction dates
            days_with_tx = [int(t.date.split("-")[2]) for t in txs if "-" in t.date]
            current_day = max(days_with_tx) if days_with_tx else 1
        else:
            current_day = 1

        days_elapsed = max(1, current_day)
        remaining_days = max(0, total_days_in_month - days_elapsed)

        # Burn rate per day
        daily_burn_rate = spent_so_far / days_elapsed
        predicted_remaining = round(daily_burn_rate * remaining_days, 2)
        projected_total = round(spent_so_far + predicted_remaining, 2)
        is_overspend = projected_total > MONTHLY_ALLOWANCE
        projected_overspend_amount = round(max(0.0, projected_total - MONTHLY_ALLOWANCE), 2)

        return {
            "month_sheet": month_sheet,
            "allowance": MONTHLY_ALLOWANCE,
            "spent_so_far": round(spent_so_far, 2),
            "balance_remaining": round(balance_remaining, 2),
            "days_elapsed": days_elapsed,
            "remaining_days": remaining_days,
            "total_days": total_days_in_month,
            "daily_burn_rate": round(daily_burn_rate, 2),
            "predicted_remaining": predicted_remaining,
            "projected_total": projected_total,
            "is_overspend": is_overspend,
            "projected_overspend_amount": projected_overspend_amount,
            "overspend_percentage": round(((projected_total - MONTHLY_ALLOWANCE) / MONTHLY_ALLOWANCE) * 100, 1) if is_overspend else 0.0,
            "alert_message": (
                f"⚠️ Overspend Warning: On course to spend ₹{projected_total:,.2f} this month, exceeding your ₹5,000 allowance by ₹{projected_overspend_amount:,.2f}!"
                if is_overspend
                else f"✓ On track! Projected month total ₹{projected_total:,.2f} within your ₹5,000 allowance."
            )
        }
