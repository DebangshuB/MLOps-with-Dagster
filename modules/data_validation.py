from decimal import Decimal, InvalidOperation

import numpy as np
from pandas_schema import Column, Schema
from pandas_schema.validation import CustomElementValidation


def check_decimal(dec):
    try:
        Decimal(dec)
    except InvalidOperation:
        return False
    return True


def check_int(num):
    try:
        int(num)
    except ValueError:
        return False
    return True


def check_string(s):
    try:
        str(s)
    except InvalidOperation:
        return False
    return True


def validate_data_(df):
    # Validation elements
    decimal_val = [
        CustomElementValidation(lambda d: check_decimal(d), 'Not Decimal')
    ]

    int_val = [
        CustomElementValidation(lambda i: check_int(i), 'Not Integer')
    ]

    null_val = [
        CustomElementValidation(lambda d: d is not np.nan, 'Field Null')
    ]

    status_val = [
        CustomElementValidation(
            lambda e: e in ["Under Construction", "Ready to move"],
            'Invalid Status'
        )
    ]

    age_val = [
        CustomElementValidation(
            lambda e: e in ["Resale", "New"],
            'Invalid Age'
        )
    ]

    unit_val = [
        CustomElementValidation(
            lambda e: e in ["L", "Cr"],
            'Invalid Unit'
        )
    ]

    # Validation schema
    schema = Schema([
        Column('BHK', int_val + null_val),
        Column('SQFT', decimal_val + null_val),
        Column('Status', status_val),
        Column('Age', age_val),
        Column('Price', decimal_val + null_val),
        Column('Unit', unit_val),
    ])

    # Apply validation
    errors = schema.validate(df)
    errors_index_rows = [e.row for e in errors]
    df_clean = df.drop(index=errors_index_rows)

    return {
        "df_clean": df_clean,
        "df_error_no": len(errors_index_rows)
    }