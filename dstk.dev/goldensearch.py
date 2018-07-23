from math import sqrt
import operator


class GoldenSearch:
    def __init__(self, func, a, d, op=operator.lt, wrapx=lambda x: x):
        self.func = func
        self.a = a
        self.d = d
        self.wrapx = wrapx
        self.op = op
        self.phi = (-1 + sqrt(5)) / 2
        self.b = self.d + self.phi * (self.a - self.d)
        self.c = self.a + self.phi * (self.d - self.a)

        self.fa, self.fb, self.fc, self.fd = map(func, [self.a, self.b, self.c, self.d])

    def step(self):
        if self.op(self.fb, self.fc):
            self.d, self.fd, self.c, self.fc = self.c, self.fc, self.b, self.fb
            self.b = self.d + self.phi * (self.a - self.d)
            self.fb = self.func(self.wrapx(self.b))
        else:
            self.a, self.fa, self.b, self.fb = self.b, self.fb, self.c, self.fc
            self.c = self.a + self.phi * (self.d - self.a)
            self.fc = self.func(self.wrapx(self.c))

        if not (self.op(self.fb, self.fa) and self.op(self.fc, self.fd)):
            chain = (self.op(self.fa, self.fb), self.op(self.fb, self.fc), self.op(self.fc, self.fd))
            if chain == (True, True, True):
                raise ValueError("Extremum outside range. Try lower range < {}".format(self.a))
            elif chain == (False, False, False):
                raise ValueError("Extremum outside range. Try upper range > {}".format(self.d))
            elif chain == (True, True, False) or chain == (False, True, True):
                raise ValueError("Opposite extremum found. Cannot determine direction.")
            raise ValueError("Multimodality found.")

        if self.op(self.fb, self.fc):
            return self.wrapx(self.b), self.fb
        else:
            return self.wrapx(self.c), self.fc


if __name__ == '__main__':
    g = GoldenSearch(lambda x: x ** 2, -10, 40)
    for i in range(30):
        print(g.step())
