"""
[user]
    Please write a Python function `data_imputation(name, addr, phone, type)` that completes the following goal:
    Given a restaurant's information, deduce the city it is located in.
    The input contains the following attributes:
    - name: str. The name of the restaurant.
    - addr: str. The address of the restaurant.
    - phone: str. The phone number of the restaurant.
    - type: str. The food type of the restaurant.
    The function is expected to output:
    - str. The city the restaurant is in.
    Examples:
    Example #0:
    Inputs:
    - name: le chardonnay (los angeles)
    - addr: 8284 melrose ave.
    - phone: 213-655-8880
    - type: french bistro
    Output:
    - los angeles
    Explanation: In this case, the city is contained in the restaurant's name. Also, phone number has area code 213 which represents Los Angeles, California.Example #1:
    Inputs:
    - name: matsuhisa
    - addr: 129 n. la cienega blvd.
    - phone: 310/659-9639
    - type: asian
    Output:
    - beverly hills
    Explanation: Phone number has area code 310 which represents Beverly Hills, California.
    The following tools could be considered:
    - python packages: You can use any python packages you want. You do not need to install but only import them before using. You can not use supervised-learning method as there is no training data. Though, you can use frozen models if you want.
    Hint: Use Regular Expressions: You can use Python's built-in `re` module to extract the city name from the restaurant's name if it is present. This approach is simple and fast, but it may not always work if the city name is not included in the restaurant's name.
    Notice that the evaluation will severely punish incorrect outputs. Thus, when the function is uncertain, please return `None` to abstain instead of returning an incorrect guess.
    The generated function should be robust, instead of only passing the provided examples. You are allowed use global variables to keep track of new incoming data.
    Please respond with the Python implementation of `data_imputation(name, addr, phone, type)` only. Please do not output any other responses or any explanations.
    Your response should be in the following format (the markdown format string should be included):
    ```python
    def data_imputation(name, addr, phone, type):
        '''Your Implementation Here.'''
    ```
"""

import re
import phonenumbers
from phonenumbers import geocoder

def data_imputation(name, addr, phone, type):
    # Extract city from name
    city_from_name = re.search(r'\((.*?)\)',name)
    if city_from_name:
        return city_from_name.group(1).lower()

    # Extract city from phone number
    try:
        phone_number = phonenumbers.parse(phone, "US")
        city_from_phone = geocoder.description_for_number(phone_number, 'en')
        if city_from_phone:
            return city_from_phone.lower()
    except:
        pass

    return None