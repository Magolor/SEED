
from seed import *

FUNCTIONS = []
# from . import data_imputation_code_V1
# FUNCTIONS.append(data_imputation_code_V1.data_imputation)

# from . import data_imputation_code_V3
# FUNCTIONS.append(data_imputation_code_V3.data_imputation)

# from . import data_imputation_code_V11
# FUNCTIONS.append(data_imputation_code_V11.data_imputation)

# from . import data_imputation_code_V13
# FUNCTIONS.append(data_imputation_code_V13.data_imputation)


def data_imputation(name, addr, phone, type):
    responses = []
    for function in FUNCTIONS:
        try:
            responses.append(
                function(
                    name=name,
addr=addr,
phone=phone,
type=type
                )
            )
        except Exception as e:
            print(e)
            responses.append(None)

    # If all responses are None, return None
    if responses == [None for _ in responses]:
        return None
    
    # If a majority of respones exists, return the majority
    hist = sorted(Counter(responses).items(),key=lambda x:x[1],reverse=True)
    if (len(hist)==1) or (len(hist)>1 and hist[0][1]>hist[1][1]):
        return hist[0][0]
    
    # If a majority of respones does not exist, return the first non-None response
    for response in responses:
        if response is not None:
            return response
