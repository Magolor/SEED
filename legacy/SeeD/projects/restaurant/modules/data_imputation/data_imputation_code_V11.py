"""
[user]
    Consider the following task:
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
    Consider the following Python function `data_imputation(name, addr, phone, type)` that is expected to complete the above task:
    ```python
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
    
        return None```
    The function seems to be incorrect:
    the function seems to be correct in its approach. it first tries to extract the city from the restaurant's name. if it fails, it then tries to extract the city from the phone number using the phonenumbers library. however, the function does not consider the address of the restaurant which could also contain the city information. also, the function does not handle the case when the phone number is not in the correct format and phonenumbers library fails to parse it.
    Hint:To fix the function, you should add a step to extract the city from the restaurant's address using regular expressions or a library like usaddress. Additionally, you should add error handling for when the phone number is not in the correct format. If the phonenumbers library fails to parse it, you can return an error message or try to extract the city from the address instead.
    Please fix the code. The fix should be robust and general, instead of only passing the provided error cases.
    Please respond with the fixed Python implementation of `data_imputation(name, addr, phone, type)` only. Please do not output any other responses or any explanations.
    Your response should be in the following format (the markdown format string should be included):
    ```python
    def data_imputation(name, addr, phone, type):
        '''Your Implementation Here.'''
    ```
"""

import re
import phonenumbers
from phonenumbers import geocoder
import usaddress

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

    # Extract city from address
    try:
        address_parts = usaddress.parse(addr)
        for part in address_parts:
            if 'PlaceName' in part:
                return part[0].lower()
    except:
        pass

    return None