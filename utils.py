import random
def generate_unique_number(length):
    min_value = 10 ** (length - 1)
    max_value = (10 ** length) - 1
    return random.randint(min_value, max_value)