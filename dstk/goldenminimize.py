from collections import namedtuple


class GoldenInconsistent(Exception):
    def __init__(self, golden_state):
        self.golden_state = golden_state
        
        
GoldenState=namedtuple("GoldenState", "a b c ya yb yc")


def golden_minimize(xs, ys=None, min_bound=True, max_bound=True):
    """
    Pass 2 or 3 initial values
    Range may extend within bounds if minimum appears at edge
    
    Example:
    def func(x):
        return x**2
    
    xs = [-10, 5]
    mini = golden_minimize(xs, [func(x) for x in xs])
    x, state = next(mini)
    for _ in range(10):
        x, state = mini.send(func(x))
        
    The case a b b a  with b < a is not solved at throws GoldenInconsistent        
    """
    if ys is None:
        ys=[]
        for x in xs:
            y=yield x, None
            ys.append(y)
    
    if len(xs) == 2:
        a, c = xs
        ya, yc = ys
        b = a + pos * (c - a)
        yb = yield b, GoldenState(a, b, c, ya, None, yc)
    else:
        a, b, c = xs
        ya, yb, yc = ys

    if min_bound is True:
        min_bound = a

    if max_bound is True:
        max_bound = c

    while 1:
        assert a < b < c, "x coordinates not ordered"
        
        golden_state = GoldenState(a, b, c, ya, yb, yc)

        d1 = b - a
        d2 = c - b
        l = c - a

        if ya < yb <= yc:  # extend region left
            if min_bound < a:
                new_x = b - d1 / pos

                if new_x < min_bound:
                    new_x = min_bound

                new_y = yield new_x, golden_state
                a, b, c = new_x, a, b
                ya, yb, yc = new_y, ya, yb
            else:
                new_x = a + pos * (b - a)
                new_y = yield new_x, golden_state
                b, c = new_x, b
                yb, yc = new_y, yb

        elif ya >= yb > yc:  # extend region right
            if c < max_bound:
                new_x = b + d2 / pos
                if new_x > max_bound:
                    new_x = max_bound

                new_y = yield new_x, golden_state
                a, b, c = b, c, new_x
                ya, yb, yc = yb, yc, new_y
            else:
                new_x = c - pos * (c - b)
                new_y = yield new_x, golden_state
                a, b = b, new_x
                ya, yb = yb, new_y

        elif ya >= yb and yb <= yc:
            if d1 < d2:
                new_x = c - l * pos
                new_y = yield new_x, golden_state
                
                if new_y > yc:
                    raise GoldenInconsistent(golden_state)
                
                if new_y < yb:
                    a, b = b, new_x
                    ya, yb = yb, new_y
                elif new_y > yb:
                    c = new_x
                    yc = new_y
                else:
                    raise StopIteration()
            else:
                new_x = a + l * pos
                new_y = yield new_x, golden_state
                
                if new_y > ya:
                    raise GoldenInconsistent(golden_state)
                
                if new_y < yb:
                    b, c = new_x, b
                    yb, yc = new_y, yb
                elif new_y > yb:
                    a = new_x
                    ya = new_y
                else:
                    raise GoldenInconsistent(golden_state)
        else:
            raise GoldenInconsistent(golden_state)


if __name__ == "__main__":
    def func(x):
        return x ** 2


    xs = [2, 5]
    mini = golden_minimize(xs, [func(x) for x in xs], min_bound=1.5)
    x = next(mini)
    for _ in range(10):
        x = mini.send(func(x))
        print(x)