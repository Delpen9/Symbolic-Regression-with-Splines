import numpy as np

class BSpline:
    def __init__(self, control_points, degree, knots):
        self.control_points = control_points
        self.degree = degree
        self.knots = knots
        self.n = control_points.shape[0] - 1
        self.d = control_points.shape[1]
        
    def __call__(self, t):
        basis = self._basis_functions(t)
        return np.dot(basis, self.control_points)
        
    def _basis_functions(self, t):
        span = self._find_span(t)
        N = np.zeros((self.degree + 1, self.d))
        N[0] = 1
        for k in range(1, self.degree + 1):
            left = np.zeros(self.d)
            right = np.zeros(self.d)
            for j in range(k):
                temp = (t - self.knots[span-k+1+j]) / (self.knots[span+j+1] - self.knots[span-k+1+j])
                left += N[j] * temp
                right += N[j] * (1-temp)
            N[k] = left + right
        return N

    def _find_span(self, t):
        if t == self.knots[-1]:
            return self.n
        low, high = self.degree, self.n + 1
        while high - low > 1:
            mid = (high + low) // 2
            if t < self.knots[mid]:
                high = mid
            else:
                low = mid
        return low
