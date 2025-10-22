import timeit
import numpy as np

setup = """\
import numpy as np
a = np.arange(1, 5)
b = np.arange(4, 8)
c = np.arange(7, 11)
"""

stmt1 = """\
a = a + b
a = a * c
"""

stmt2 = """\
a += b
a *= c
"""

print(timeit.timeit(
    setup=setup,
    stmt=stmt1,
    number=10000000
))

print(timeit.timeit(
    setup=setup,
    stmt=stmt2,
    number=10000000
))