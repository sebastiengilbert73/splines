import matplotlib.pyplot as plt
import logging
import sys
sys.path.append("..")
import src.splines.spline_1d as spline_1d
import torch
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("test_batch_evaluate.main()")

    xys = np.array([[-0.2, 1], [0, 0.8], [0.1, 0.3], [0.3, -0.2], [0.5, 0], [0.7, 1], [0.8, 0.7]])
    cubic_spline = spline_1d.CubicSpline(xys, boundary_condition='2nd_derivative_0')

    xs = np.arange(-0.3, 1.005, 0.01)
    y_pred = []
    for x in xs:
        print(x)
        y_pred.append(cubic_spline.evaluate(x))

    x_tsr = torch.from_numpy(xs).unsqueeze(1)  # (B, 1)
    batch_y_tsr = cubic_spline.batch_evaluate(x_tsr)

    fig, ax = plt.subplots(1)
    ax.plot(xs, y_pred, label='Cubic spline, evaluate()', linewidth=3)
    ax.plot(xs, batch_y_tsr.squeeze().detach().numpy(), label='Cubic spline, batch_evaluate()', linestyle='--')
    ax.scatter([x for x, _ in xys], [y for _, y in xys], c='red', marker='o', label="Knots")
    ax.grid(True)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()