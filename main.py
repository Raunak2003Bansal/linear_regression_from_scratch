import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_loss(m: float, b: float, points: pd.DataFrame) -> float:
    """Calculate the mean squared error for a line and a set of points."""
    total_loss = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_loss += (y - (m * x + b)) ** 2
    return total_loss / float(len(points))


def gradient_descent(
    m: float, b: float, points: pd.DataFrame, L: float
) -> tuple[float, float]:
    m_gradient = 0
    b_gradient = 0

    n = len(points)
    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        m_gradient += -(2 / n) * x * (y - (m * x + b))
        b_gradient += -(2 / n) * (y - (m * x + b))

    new_m = m - L * m_gradient
    new_b = b - L * b_gradient
    return new_m, new_b


def optimize(
    points: pd.DataFrame,
    m: float,
    b: float,
    L: float,
    num_iterations: int,
) -> tuple[float, float]:
    for i in range(num_iterations):
        if i % 100 == 0:
            print(
                f"Iteration {i}: m = {m}, b = {b}, "
                f"loss = {np.sqrt(calculate_loss(m, b, points))}"
            )
        m, b = gradient_descent(m, b, points, L)
    return m, b


def main() -> None:
    df = pd.read_csv("Exam_Score_Prediction.csv")
    df = df[["study_hours", "exam_score"]].copy()
    df = df.rename(columns={"study_hours": "studytime", "exam_score": "score"})
    print(df)
    df = df.sample(1000, random_state=42)

    sample_loss = calculate_loss(0.5, 0.5, df)
    rmse = np.sqrt(sample_loss)
    print(sample_loss)
    print(rmse)

    m, b = optimize(df, 0.5, 0.5, 0.01, 1000)
    print(m, b)

    plt.scatter(df.studytime, df.score, color="blue", label="Data Points")
    y_pred = m * df.studytime + b
    plt.plot(df.studytime, y_pred, color="red", label="Regression Line")
    plt.xlabel("X Study Time (hours)")
    plt.ylabel("Y Exam Score")
    plt.title("Simple Linear Regression Fit")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    rmse = np.sqrt(calculate_loss(m, b, df))
    print(f"Root Mean Squared Error (RMSE): {rmse}")


if __name__ == "__main__":
    main()
