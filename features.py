"""Shared feature definitions so training and serving can never drift apart."""

EDU_MAP = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}

# Housing_Status is one-hot encoded with Mortgage as the baseline category.
HOUSING_MAP = {
    "Mortgage": (0, 0),
    "Own": (1, 0),
    "Rent": (0, 1),
}

FEATURE_COLS = [
    "Age", "Income", "Loan_Amount", "Credit_Score",
    "Employment_Years", "Education_Level",
    "Housing_Status_Own", "Housing_Status_Rent",
]

# Manual safety rule applied on top of the model (see README).
RULE_MAX_CREDIT_SCORE = 500
RULE_LOAN_TO_INCOME = 5
RULE_MIN_EMPLOYMENT_YEARS = 1
RULE_MIN_DEFAULT_PROB = 90.0


def rule_triggered(credit_score, loan_amount, income, employment_years) -> bool:
    return (
        credit_score < RULE_MAX_CREDIT_SCORE
        and loan_amount > income * RULE_LOAN_TO_INCOME
        and employment_years < RULE_MIN_EMPLOYMENT_YEARS
    )
