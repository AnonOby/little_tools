Create a file named `README.md` with the following content:

```markdown
# Multi-Function Plotter

A simple interactive Python tool that lets you plot multiple mathematical functions on the same graph. Enter expressions like `sin(x)`, `x**2`, `log(x)`, etc., and see them overlaid for comparison.

## Features

- Accepts any number of functions (enter one per line, empty line to finish).
- Uses a safe evaluation environment (no dangerous built‑ins allowed).
- Plots all functions together with a legend and grid.
- Automatically handles a wide range of mathematical functions (trigonometric, exponential, logarithmic, etc.).

## Requirements

- Python 3.6 or higher
- `matplotlib` and `numpy`

Install them with:

```bash
pip install matplotlib numpy
```

## Usage

### Run the script directly
```bash
python plot_function.py
```

### Use the included batch file (Windows)
Double‑click `plot_function.bat`. It will install the required packages (if missing) and start the plotter.

### Interactive session
1. Enter a function of `x` (e.g., `x**2`, `sin(x)`, `exp(-x)*cos(x)`).
2. Press Enter to add another function.
3. When done, press Enter on an empty line to generate the plot.

Example:
```
f(x) = x**2
  -> Added 'x**2'
f(x) = sin(x)
  -> Added 'sin(x)'
f(x) = 
```

The plot window will show both curves.

## Supported Functions

The expression can use:
- Basic arithmetic: `+`, `-`, `*`, `/`, `**` (power)
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Hyperbolic: `sinh`, `cosh`, `tanh`
- Exponential/log: `exp`, `log` (natural log), `log10`
- Other: `sqrt`, `abs`, `sign`, `ceil`, `floor`
- Constants: `pi`, `e`
- Numpy functions: `maximum`, `minimum`, `mod`, `power`, `arctan2`, etc.

> ⚠️ Expressions are evaluated with `eval()` in a restricted namespace. Dangerous patterns (e.g., `__`, `exec`, `import`) are blocked.

## License

Feel free to use, modify, and distribute this script for any purpose.
