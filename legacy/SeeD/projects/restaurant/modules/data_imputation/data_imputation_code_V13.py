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
    from geopy.geocoders import Nominatim
    
    def data_imputation(name, addr, phone, type):
        # Extract city from restaurant name
        city_in_name = re.search(r'\((.*?)\)', name)
        if city_in_name:
            return city_in_name.group(1).lower()
    
        # Extract city from phone number
        try:
            parsed_phone = phonenumbers.parse(phone, 'US')
            city_in_phone = phonenumbers.geocoder.description_for_number(parsed_phone, 'en')
            if city_in_phone:
                return city_in_phone.lower()
        except:
            pass
    
        # Extract city from address using Geocoding API
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.geocode(addr)
        if location:
            return location.address.split(',')[-3].strip().lower()
    
        return None```
    The function seems to be incorrect:
    the function seems to be logically correct. it first tries to extract the city from the restaurant's name. if it fails, it tries to extract the city from the phone number. if it still fails, it uses the geocoding api to extract the city from the address. however, the function assumes that the city is always the third last element in the address, which may not always be the case. also, the function does not handle the case where the phone number does not have an area code.
    Hint:The function should be modified to handle the case where the phone number does not have an area code. This can be done by checking if the parsed phone number has an area code before trying to extract the city from it. For the address, instead of assuming the city is always the third last element, the function should parse the address and look for the city in it. This can be done by using a library like `usaddress` which can parse US addresses and return a dictionary of address components. The function should then check if the 'City' key is in the dictionary and return its value if it is.
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
import usaddress
from geopy.geocoders import Nominatim

def data_imputation(name, addr, phone, type):
    # Extract city from restaurant name
    city_in_name = re.search(r'\((.*?)\)', name)
    if city_in_name:
        return city_in_name.group(1).lower()

    # Extract city from phone number
    try:
        parsed_phone = phonenumbers.parse(phone, 'US')
        if parsed_phone.country_code == 1 and parsed_phone.national_number:
            city_in_phone = phonenumbers.geocoder.description_for_number(parsed_phone, 'en')
            if city_in_phone:
                return city_in_phone.lower()
    except:
        pass

    # Extract city from address using Geocoding API
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(addr)
    if location:
        address_components = usaddress.parse(location.address)
        for component in address_components:
            if 'PlaceName' in component:
                return component[0].lower()

    return None