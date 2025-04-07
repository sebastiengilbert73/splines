import logging
import sys
sys.path.append("..")
import src.splines.spline_1d as spline_1d
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("test_spline1d.py")

    np.random.seed(0)

    xys = np.array([[-0.2, 1], [0, 0.8], [0.1, 0.3], [0.3, -0.2], [0.5, 0], [0.7, 1], [0.8, 0.7]])
    spline1 = spline_1d.LinearSpline(xys)
    spline2 = spline_1d.QuadraticSpline(xys, boundary_condition='right_2nd_derivative_0')
    spline3 = spline_1d.CubicSpline(xys, boundary_condition='2nd_derivative_0')

    xs = np.arange(-0.3, 1.005, 0.01)
    y_pred_1 = []
    y_pred_2 = []
    y_pred_3 = []
    for x in xs:
        print(x)
        y_pred_1.append(spline1.evaluate(x))
        y_pred_2.append(spline2.evaluate(x))
        y_pred_3.append(spline3.evaluate(x))

    fig, ax = plt.subplots(1)
    ax.plot(xs, y_pred_1, label='Linear spline')
    ax.plot(xs, y_pred_2, label='Quadratic spline')
    ax.plot(xs, y_pred_3, label='Cubic spline')
    ax.scatter([x for x, _ in xys], [y for _, y in xys], c='red', marker='o', label="Knots")
    ax.grid(True)
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()