"""
plot_function.py
----------------
Extended version: supports plotting multiple functions, solving equations,
numerical differentiation, and evaluating at points.
Now uses sympy for exact symbolic solving and derivative expressions.
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.optimize as opt
import warnings

# Try to import sympy for symbolic capabilities
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    warnings.warn("Sympy not installed. Symbolic solving and derivative expressions will not be available. Install with 'pip install sympy'")

# Safe namespace for evaluating user expressions (numeric)
safe_dict = {
    'x': None,
    'np': np,
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'exp': np.exp, 'log': np.log, 'log10': np.log10,
    'sqrt': np.sqrt, 'abs': np.abs, 'sign': np.sign,
    'pi': np.pi, 'e': np.e, 'arctan2': np.arctan2,
    'degrees': np.degrees, 'radians': np.radians,
    'ceil': np.ceil, 'floor': np.floor,
    'mod': np.mod, 'power': np.power,
    'maximum': np.maximum, 'minimum': np.minimum,
}

def is_safe_expression(expr):
    """Reject expressions containing double underscores or dangerous keywords."""
    dangerous = re.compile(r'(__|exec|eval|import|compile|globals|locals|open|file)')
    return not dangerous.search(expr)

def evaluate_expression(expr, x_array):
    """Evaluate the expression safely for the given x array."""
    safe_dict['x'] = x_array
    try:
        return eval(expr, safe_dict)
    except Exception as e:
        print(f"Error: {e}")
        return None

def symbolic_derivative(expr_str, order=1):
    """Return the symbolic derivative of order 'order' as a string."""
    if not SYMPY_AVAILABLE:
        return None
    try:
        # Remove np. prefix
        expr_clean = re.sub(r'\bnp\.', '', expr_str)
        x_sym = sp.Symbol('x')
        sympy_funcs = {
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'exp': sp.exp, 'log': sp.log, 'log10': lambda x: sp.log(x, 10),
            'sqrt': sp.sqrt, 'abs': sp.Abs, 'sign': sp.sign,
            'pi': sp.pi, 'e': sp.E, 'E': sp.E,
            'arctan2': sp.atan2,
            'degrees': lambda x: x * 180 / sp.pi,
            'radians': lambda x: x * sp.pi / 180,
            'ceil': sp.ceiling, 'floor': sp.floor,
            'mod': sp.Mod, 'power': sp.Pow,
            'maximum': sp.Max, 'minimum': sp.Min,
        }
        expr_sym = sp.sympify(expr_clean, locals=sympy_funcs)
        deriv = sp.diff(expr_sym, x_sym, order)
        return str(deriv)
    except Exception as e:
        print(f"Symbolic derivative failed: {e}")
        return None

def compute_derivative_numeric(expr, x_grid, order):
    """Compute the numerical derivative of the given order (1–5)."""
    safe_dict['x'] = x_grid
    y = eval(expr, safe_dict)
    for _ in range(order):
        y = np.gradient(y, x_grid)
    return y

def plot_derivatives(expr, x_grid, orders):
    """Plot the original function and its derivatives up to the given orders."""
    print("Generating derivative plot...")
    plt.figure(figsize=(10,6))
    safe_dict['x'] = x_grid
    y = eval(expr, safe_dict)
    plt.plot(x_grid, y, label=f'f(x) = {expr}', linewidth=2)

    colors = plt.cm.tab10(np.linspace(0,1,len(orders)))
    for order, color in zip(orders, colors):
        if SYMPY_AVAILABLE:
            deriv_expr = symbolic_derivative(expr, order)
            if deriv_expr:
                print(f"Derivative of order {order}: {deriv_expr}")
            label = f"f^{order}'(x)"
            if deriv_expr:
                label += f" = {deriv_expr}"
        else:
            label = f"f^{order}'(x) (numeric)"
        y_deriv = compute_derivative_numeric(expr, x_grid, order)
        plt.plot(x_grid, y_deriv, '--', color=color, label=label, linewidth=1.5)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.title(f"Function and its derivatives up to order {max(orders)}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    print("Derivative plot closed. Returning to menu.")

def solve_equation_symbolic(expr1, expr2=None, target=0):
    """
    Solve f(x)=target or f(x)=g(x) symbolically using sympy.
    Returns a string representation of the solution set.
    """
    if not SYMPY_AVAILABLE:
        return None
    try:
        # Remove np. prefix
        expr1_clean = re.sub(r'\bnp\.', '', expr1)
        if expr2:
            expr2_clean = re.sub(r'\bnp\.', '', expr2)
        else:
            expr2_clean = None
        x_sym = sp.Symbol('x')
        sympy_funcs = {
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'exp': sp.exp, 'log': sp.log, 'log10': lambda x: sp.log(x, 10),
            'sqrt': sp.sqrt, 'abs': sp.Abs, 'sign': sp.sign,
            'pi': sp.pi, 'e': sp.E, 'E': sp.E,
            'arctan2': sp.atan2,
            'degrees': lambda x: x * 180 / sp.pi,
            'radians': lambda x: x * sp.pi / 180,
            'ceil': sp.ceiling, 'floor': sp.floor,
            'mod': sp.Mod, 'power': sp.Pow,
            'maximum': sp.Max, 'minimum': sp.Min,
        }
        f1_sym = sp.sympify(expr1_clean, locals=sympy_funcs)
        if expr2_clean is None:
            equation = sp.Eq(f1_sym, target)
        else:
            f2_sym = sp.sympify(expr2_clean, locals=sympy_funcs)
            equation = sp.Eq(f1_sym, f2_sym)
        sol_set = sp.solveset(equation, x_sym, domain=sp.S.Reals)
        return sol_set
    except Exception as e:
        print(f"Symbolic solving failed: {e}")
        return None

def solve_equation_numeric(expr, x_range=None, target=0, other_expr=None):
    """
    Numerically solve f(x)=target or f(x)=g(x) by scanning the given range
    and using fsolve from multiple starting points.
    Returns a list of roots.
    """
    if x_range is None:
        x_range = (-500, 500)
    x_min, x_max = x_range
    # Create a dense grid to detect sign changes
    x_grid = np.linspace(x_min, x_max, 10000)
    def func(x):
        safe_dict['x'] = np.array([x])
        if other_expr is None:
            return eval(expr, safe_dict)[0] - target
        else:
            f1 = eval(expr, safe_dict)[0]
            f2 = eval(other_expr, safe_dict)[0]
            return f1 - f2

    # Evaluate on grid to find intervals where sign changes
    y_grid = np.array([func(x) for x in x_grid])
    sign_changes = np.where(np.diff(np.sign(y_grid)))[0]
    roots = []
    for idx in sign_changes:
        x0 = x_grid[idx]
        x1 = x_grid[idx+1]
        # Use brentq which requires a bracket
        try:
            root = opt.brentq(func, x0, x1, xtol=1e-12)
            if not any(abs(root - r) < 1e-8 for r in roots):
                roots.append(root)
        except ValueError:
            # If brentq fails, try fsolve
            try:
                root = opt.fsolve(func, x0)[0]
                if not any(abs(root - r) < 1e-8 for r in roots):
                    roots.append(root)
            except:
                pass
    return roots

def solve_equation(expr1, expr2=None, target=0, x_range=None):
    """
    High-level solving function: tries symbolic first, then numeric.
    Prints the result clearly.
    """
    if SYMPY_AVAILABLE:
        sol_set = solve_equation_symbolic(expr1, expr2, target)
        if sol_set is not None:
            print("Solution set:")
            if sol_set == sp.EmptySet:
                print("No real solutions.")
            elif sol_set == sp.S.Reals:
                print("All real numbers are solutions.")
            else:
                # Convert to string and print nicely
                print(sol_set)
            return

    # Fallback to numeric
    print("Symbolic solving failed or not available. Using numeric method (may not find all roots).")
    roots = solve_equation_numeric(expr1, x_range, target, expr2)
    if roots:
        print(f"Found {len(roots)} root(s):")
        for r in roots:
            safe_dict['x'] = np.array([r])
            val = eval(expr1, safe_dict)[0]
            if expr2 is None:
                print(f"x = {r:.8f}  (f(x) = {val:.8f})")
            else:
                g_val = eval(expr2, safe_dict)[0]
                print(f"x = {r:.8f}  (f={val:.8f}, g={g_val:.8f})")
    else:
        print("No roots found in the scanned range. Try a different range or use symbolic solving (install sympy).")

def main():
    print("=============================================")
    print("   Multi-Function Plotter (Extended)")
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
            if not expressions:
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

    # Additional interactive menu
    while True:
        print("\n====== Additional Operations ======")
        print("1. Solve equation (f(x)=c or f(x)=g(x))")
        print("2. Compute derivatives and plot (1–5 orders)")
        print("3. Evaluate function at a point")
        print("4. Exit")
        choice = input("Please choose (1-4): ").strip()
        if choice == '4':
            break

        if choice == '1':
            print("Available functions:")
            for i, expr in enumerate(expressions, 1):
                print(f"{i}. {expr}")
            idx = int(input("Select the first function number: ")) - 1
            expr1 = expressions[idx]
            eq_type = input("Equation type: (1) f(x)=c   (2) f(x)=g(x): ")
            if eq_type == '1':
                target_str = input("Enter target value c (default 0): ").strip()
                target = float(target_str) if target_str else 0.0
                solve_equation(expr1, target=target)
            elif eq_type == '2':
                idx2 = int(input("Select the second function number: ")) - 1
                expr2 = expressions[idx2]
                solve_equation(expr1, expr2=expr2)
            else:
                print("Invalid option")

        elif choice == '2':
            print("Available functions:")
            for i, expr in enumerate(expressions, 1):
                print(f"{i}. {expr}")
            idx = int(input("Select a function number: ")) - 1
            expr = expressions[idx]
            orders_str = input("Enter derivative orders (comma separated, e.g., 1,2,3) [1-5]: ")
            orders = [int(o.strip()) for o in orders_str.split(',') if 1 <= int(o.strip()) <= 5]
            if orders:
                print("Derivative orders selected:", orders)
                if not SYMPY_AVAILABLE:
                    print("Note: sympy not installed. Derivative expressions will be numeric only.")
                plot_derivatives(expr, x, orders)
            else:
                print("Invalid orders, only 1-5 are supported")

        elif choice == '3':
            print("Available functions:")
            for i, expr in enumerate(expressions, 1):
                print(f"{i}. {expr}")
            idx = int(input("Select a function number: ")) - 1
            expr = expressions[idx]
            try:
                x0 = float(input("Enter x value: "))
            except:
                print("Invalid input")
                continue
            safe_dict['x'] = np.array([x0])
            y0 = eval(expr, safe_dict)[0]
            print(f"f({x0}) = {y0:.6f}")

        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
