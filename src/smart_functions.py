class SmartFunction:
    def __init__(self):
        pass 

    def infer_schema(self, input_dictionary):
        """Extracts keys with numeric values (bool, int, float)."""
        return [k for k, v in input_dictionary.items() if isinstance(v, (bool, int, float))]

    def assert_numerics_are_equal(self, original_dictionary, comparison_dictionary):
        """Compares numeric values between two dictionaries."""
        try:
            numeric_fields = self.infer_schema(original_dictionary)
            for key in numeric_fields:
                if key not in comparison_dictionary or original_dictionary[key] != comparison_dictionary[key]:
                    return False  # If any numeric field is different, return False
            return True  # If all numeric fields match, return True
        except Exception as e:
            print(f"Error: {e}")
            return False
        
    def is_improvement_required(self, orginal_dictionary, comparison_dictionary):
        """Calls previous methods assuming we are parsing output of the assessor agent."""
        try:
            comparison_dictionary = comparison_dictionary['improvement']
            is_equal = self.assert_numerics_are_equal(orginal_dictionary, comparison_dictionary)
            if is_equal:
                return False
            else:
                return True
        except:
            print('Coult not obtain dictionary')
            return None
        

# Instantiate and run on test cases
original_dictionary = {
    'name' : 'Joy',
    'age' : 28,
    'is_engineer' : True,
    'is_overly_serious' : False,
    'favourite_number' : 8.8
}

comparison_dictionary = {
    'name': 'Alice',
    'age': 28,
    'is_engineer': True,
    'is_overly_serious': False,
    'favourite_number': 8.8
}

false_dictionary = original_dictionary.copy()
false_dictionary['age'] = 29 # change one numeric value to know you yield a false value

if __name__ == "__main__":
    smart_function = SmartFunction()
    numeric_keys = smart_function.infer_schema(original_dictionary)
    is_equal = smart_function.assert_numerics_are_equal(original_dictionary, comparison_dictionary)
    print(numeric_keys)
    print(is_equal)

    false_is_equal = smart_function.assert_numerics_are_equal(original_dictionary, false_dictionary)
    print(false_is_equal)