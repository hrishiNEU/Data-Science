import numpy as np
def interquartile(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    Q1_pos = (n+1) // 4
    Q1 = sorted_data[Q1_pos-1]

    Q3_pos = (n+1) * 3 // 4
    Q3 = sorted_data[Q3_pos-1]

    print("Q1 is %d "%Q1)
    print("Q3 is %d "%Q3)

    IQR = Q3 - Q1
    print("Value of IQR is %d "%IQR)

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")

data = [15, 175, 80, 34, 23, 12, -95, 74, 56, 9, 65]
interquartile(data)