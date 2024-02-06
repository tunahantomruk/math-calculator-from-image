import re

# Tokenize the input string using regular expressions
def tokenize(expression):
    return re.findall(r'\d+|\D', expression)

# Convert the symbol to the corresponding arithmetic operator
operator = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x / y,
    'x': lambda x, y: x * y
    #'(': lambda x, y: x * y  2*3(5-1) Result: 29
}

# Define parsing functions for recursive descent parsing
def parse_number(tokens):
    return int(tokens.pop(0))

def parse_factor(tokens):
    if tokens[0].isdigit():
        return parse_number(tokens)
    elif tokens[0] == '(':
        tokens.pop(0)
        result = parse_expression(tokens)
        if tokens[0] == ')':
            tokens.pop(0)
        return result

    raise ValueError("Invalid expression")

def parse_term(tokens):
    left = parse_factor(tokens)

    while tokens and tokens[0] in ('*', '/','x'):
        operator_token = tokens.pop(0)
        right = parse_factor(tokens)
        left = operator[operator_token](left, right)

    return left

def parse_expression(tokens):
    left = parse_term(tokens)

    while tokens and tokens[0] in ('+', '-'):
        operator_token = tokens.pop(0)
        right = parse_term(tokens)
        left = operator[operator_token](left, right)

    return left

# Evaluate the expression using the parsing functions
expression = input("Enter an arithmetic expression: ")
tokens = tokenize(expression)

# Perform the calculations if tokens are present
if tokens:
    if tokens[-2] == '=' and tokens[-1].isdigit():
        inputResult = int(tokens[-1])
        print("input result:",inputResult)
        tokens = tokens[:-2]  # Remove the '= number' part
    else:
        inputResult = None

    result = parse_expression(tokens)
    print("Result:", result)

    if inputResult is not None:
        if result == inputResult:
            print("The result matches the expected result.")
        else:
            print("The result does not match the expected result.")
else:
    print("Invalid input. Please enter a valid arithmetic expression.")
