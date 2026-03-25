"""
plot_function.py
----------------
A script that accepts multiple mathematical expressions in x and plots them
on the same graph for comparison. The user can keep entering functions until
an empty input is given. All curves are displayed together with automatic
coloring and a legend.
"""

import matplotlib.pyplot as plt
import numpy as np
import re

# Safe namespace for evaluating user expressions
safe_dict = {
    'x': None,
    'np': np,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'exp': np.exp,
    'log': np.log,
    'log10': np.log10,
    'sqrt': np.sqrt,
    'abs': np.abs,
    'sign': np.sign,
    'pi': np.pi,
    'e': np.e,
    'arctan2': np.arctan2,
    'degrees': np.degrees,
    'radians': np.radians,
    'ceil': np.ceil,
    'floor': np.floor,
    'mod': np.mod,
    'power': np.power,
    'maximum': np.maximum,
    'minimum': np.minimum,
}

def is_safe_expression(expr):
    """Reject expressions that contain double underscores or dangerous words."""
    # Blacklist: any double underscore, or known dangerous names
    dangerous = re.compile(r'(__|exec|eval|import|compile|globals|locals|open|file)')
    if dangerous.search(expr):
        return False
    return True

def evaluate_expression(expr, x_array):
    """Evaluate the user's expression safely for the given x array."""
    # Temporarily set x to the array
    safe_dict['x'] = x_array
    try:
        # Use safe_dict as the only namespace; built-ins are not available
        y = eval(expr, safe_dict)
        return y
    except Exception as e:
        print(f"Error evaluating expression '{expr}': {e}")
        return None

def main():
    print("=============================================")
    print("   Multi-Function Plotter")
    print("=============================================")
    print("Enter functions of x (e.g., x**2, sin(x)).")
    print("Press Enter on an empty line to finish and plot.")
    print("All functions will be drawn on the same graph.")
    print("---------------------------------------------")

    # Define default x range and resolution
    x_min, x_max = -500.0, 500.0
    num_points = 100000
    x = np.linspace(x_min, x_max, num_points)

    expressions = []          # store raw expressions
    curves = []               # store (expr, y) pairs

    while True:
        expr = input("f(x) = ").strip()
        if expr == "":
            if len(expressions) == 0:
                print("No functions entered. Exiting.")
                return
            break

        # Basic safety check
        if not is_safe_expression(expr):
            print("  -> Skipped (expression contains unsafe pattern)")
            continue

        # Evaluate the expression
        y = evaluate_expression(expr, x)
        if y is not None:
            expressions.append(expr)
            curves.append((expr, y))
            print(f"  -> Added '{expr}'")
        else:
            print("  -> Skipped (invalid expression)")

    # Plot all accumulated curves
    plt.figure(figsize=(10, 6))
    for expr, y in curves:
        plt.plot(x, y, linewidth=1.5, label=f"f(x) = {expr}")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Multiple Function Plot")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
