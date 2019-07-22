import random


# http://math.stackexchange.com/questions/124408/finding-a-primitive-root-of-a-prime-number
# https://github.com/cokiencoke/primitive-root/blob/master/primitive_root.py
# TODO: calc primitive root; write sklearn transformer


def category_code(n, i, primroot1, primroot2):
    """
    n needs to be a prime number and primroot1, primroot2 should be two primitive roots of it
    """
    code = {}
    val = pow(primroot1, i, n)
    for j in range(n - 1):
        code[j if j < n // 2 else n + n // 2 - 2 - j] = val
        val = val * primroot2 % n
    return code


def category_codes(n, primroot1, primroot2, sample_num=None):
    result = []

    if sample_num is None:
        sample_values = range(n // 2)
    else:
        sample_values = random.sample(range(n // 2), sample_num)

    for i in sample_values:
        result.append(category_code(n, i, primroot1, primroot2))

    return result


if __name__ == "__main__":
    from pprint import pprint

    pprint(category_codes(11, 2, 6))
