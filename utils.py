import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mapping_emp_length = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10
}

mapping_grade = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

mapping_subgrade = {
 'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5,
 'B1': 6, 'B2': 7, 'B3': 8, 'B4': 9, 'B5': 10,
 'C1': 11, 'C2': 12, 'C3': 13, 'C4': 14, 'C5': 15,
 'D1': 16, 'D2': 17, 'D3': 18, 'D4': 19, 'D5': 20,
 'E1': 21, 'E2': 22, 'E3': 23, 'E4': 24, 'E5': 25,
 'F1': 26, 'F2': 27, 'F3': 28, 'F4': 29, 'F5': 30,
 'G1': 31, 'G2': 32, 'G3': 33, 'G4': 34, 'G5': 35
}

def ordinal_encode(df, drop=False):
    df = df.copy()

    df["subgrade_encoded"] = df["sub_grade"].map(mapping_subgrade)
    df["grade_encoded"] = df["grade"].map(mapping_grade)
    df["emp_length_encoded"] = df["emp_length"].map(mapping_emp_length)

    # is it needed ? seems like mixed effect if we keep it
    if drop: 
        df = df.drop(columns=["sub_grade", "grade", "emp_length"])
    return df

class NamedFunctionTransformer(FunctionTransformer):
    def get_feature_names_out(self, input_features=None):
        return input_features  # just pass through