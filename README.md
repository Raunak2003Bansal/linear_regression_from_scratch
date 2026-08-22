# Linear Regression From Scratch

A small educational machine-learning project that predicts exam scores from study hours using simple linear regression implemented with Python, Pandas, NumPy, and Matplotlib.

The training algorithm is written from scratch with gradient descent, so the project shows the mechanics behind fitting a regression line without using a machine-learning framework.

## What This Project Does

- Loads exam-score data from a CSV file
- Selects study hours and exam score as the input and target
- Samples 1,000 rows for training
- Calculates mean squared error and RMSE
- Optimizes the slope and intercept using gradient descent
- Displays the fitted regression line with the data points

## Requirements

- Python 3.13 or newer
- [`uv`](https://docs.astral.sh/uv/)

## Setup

Clone the repository and open its directory:

```bash
git clone https://github.com/Raunak2003Bansal/linear_regression_from_scratch.git
cd linear_regression_from_scratch
```

Create or synchronize the project environment with `uv`:

```bash
uv sync
```

`uv sync` creates or updates the local `.venv` environment from `pyproject.toml` and `uv.lock`.

## Run the Model

Run the Python implementation through the managed environment:

```bash
uv run python main.py
```

The command prints the initial loss, training progress, learned parameters, and final RMSE. It also opens a chart showing the regression line.

The script expects `Exam_Score_Prediction.csv` to be in the project root.

## Dataset

The source CSV must contain these columns:

| Column | Description |
| --- | --- |
| `study_hours` | Number of hours spent studying |
| `exam_score` | Student's exam score |

The script renames these columns internally to `studytime` and `score`.

## How It Works

The model uses the line:

```text
y = m x + b
```

where `m` is the slope and `b` is the intercept. For each iteration, the gradients of the mean squared error are calculated and used to update both parameters:

```text
m = m - learning_rate * m_gradient
b = b - learning_rate * b_gradient
```

The current configuration starts with `m = 0.5` and `b = 0.5`, uses a learning rate of `0.01`, and runs for 1,000 iterations.

## Project Structure

```text
.
├── Exam_Score_Prediction.csv  # Training data
├── code.ipynb                 # Notebook version of the workflow
├── main.py                    # Standalone Python implementation
├── pyproject.toml             # Project metadata and dependencies
├── requirements.txt           # Dependency list
└── uv.lock                    # Locked uv dependency versions
```

## Learning Goals

This project is useful for understanding:

- How a linear model represents a relationship between two variables
- How mean squared error measures prediction error
- How gradients adjust model parameters
- How the learning rate affects optimization
- How to visualize a fitted model against observed data

## License

See [LICENSE](LICENSE) for license details.
