from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator
from .config import CATEGORIES


class TransactionCreate(BaseModel):
    amount: float = Field(..., gt=0, description="Amount debited in Indian Rupees (₹)")
    category: str = Field(..., description="Expense category")
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format (defaults to current date)")
    time: Optional[str] = Field(None, description="Time in HH:MM format (defaults to current time)")
    month_sheet: Optional[str] = Field(None, description="Month sheet name (e.g. Apr_2029)")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        clean = v.strip().title()
        # Handle case sensitivity
        matched = next((c for c in CATEGORIES if c.lower() == clean.lower()), None)
        if not matched:
            raise ValueError(f"Category '{v}' is invalid. Allowed: {', '.join(CATEGORIES)}")
        return matched

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Amount must be greater than zero.")
        if v > 100000:
            raise ValueError("Amount exceeds realistic student transaction limit (₹1,00,000).")
        return round(v, 2)


class TransactionResponse(BaseModel):
    id: int
    month_sheet: str
    date: str
    time: str
    amount_debited: float
    category: str
    amount_credited: float
    remaining_balance: float
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class CategorySuggestionCandidate(BaseModel):
    category: str
    probability: float
    confidence_percent: str


class CategorySuggestionResponse(BaseModel):
    suggested_category: str
    probability: float
    confidence_percent: str
    candidates: List[CategorySuggestionCandidate]
    model_accuracy: float
    disclaimer: str


class ExpectedAmountResponse(BaseModel):
    category: str
    expected_amount: float
    formatted_expected: str


class DifferenceResponse(BaseModel):
    category: str
    actual_amount: float
    expected_amount: float
    difference: float
    status: str
    summary_text: str


class CategorySpendingItem(BaseModel):
    category: str
    amount: float
    count: int
    percentage: float
    expected_average: float


class DailyTrendItem(BaseModel):
    date: str
    day: int
    daily_amount: float
    cumulative_amount: float


class MonthSummaryResponse(BaseModel):
    month_sheet: str
    allowance: float
    spent_so_far: float
    balance_remaining: float
    days_elapsed: int
    remaining_days: int
    total_days: int
    daily_burn_rate: float
    predicted_remaining: float
    projected_total: float
    is_overspend: bool
    projected_overspend_amount: float
    overspend_percentage: float
    alert_message: str
    category_spending: List[CategorySpendingItem]
    daily_trend: List[DailyTrendItem]


class ModelBenchmarkItem(BaseModel):
    model_name: str
    r2_score: float
    rmse: float
    mae: float
    is_primary: bool
    notes: Optional[str] = None


class AboutResponse(BaseModel):
    held_out_sheet: str
    total_transactions: int
    train_count: int
    test_count: int
    benchmarks: List[ModelBenchmarkItem]
    classifier_accuracy: float
    classifier_notes: str
    category_averages: Dict[str, float]
    selected_model_reason: str
