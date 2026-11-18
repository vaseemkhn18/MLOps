
import pandas as pd

def test_dataset_columns():

    df = pd.read_csv(
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    required_columns = [
        "customerID",
        "gender",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn"
    ]

    for column in required_columns:

        assert column in df.columns
