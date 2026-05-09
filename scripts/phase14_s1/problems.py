"""Phase 14 S1: 25 HumanEval-style problems for Qwen substrate
variance measurement.

Convention (mirrors Phase 9 S5):
  - prompt: function signature + docstring + `return ` (single-line
    completion expected)
  - suffix: 1-3 assert statements that exercise the function
  - solution must fit on a single line (truncate at \\n during gen)

Difficulty mix:
  - 12 likely warm-start problems (0.5B base hits ≥ 1/8 in S5-like
    runs)
  - 8 medium (occasional pass)
  - 5 cold-start (likely 0/8 — keeps headroom for self-improve)
"""

CHALLENGES = [
    # ===== 6 from Phase 9 S5 (continuity with previous measurements) =====
    {"name": "equals_5", "prompt": "def f():\n    return ",
     "suffix": "\n\nassert f() == 5\n"},
    {"name": "equals_14_via_doubling", "prompt": "def f():\n    return 2 * (",
     "suffix": ")\n\nassert f() == 14\n"},
    {"name": "len_5_string", "prompt": "s = ",
     "suffix": "\n\nassert isinstance(s, str) and len(s) == 5\n"},
    {"name": "two_plus_to_5", "prompt": "x = 2 + ",
     "suffix": "\nassert x == 5\n"},
    {"name": "ten_minus_to_3", "prompt": "x = 10 - ",
     "suffix": "\nassert x == 3\n"},
    {"name": "two_pow_to_8", "prompt": "x = 2 ** ",
     "suffix": "\nassert x == 8\n"},

    # ===== 5 from Phase 9 S5 HumanEval-style =====
    {
        "name": "is_even",
        "prompt": 'def is_even(n):\n    """Return True if n is even."""\n    return ',
        "suffix": "\n\nassert is_even(2) == True\nassert is_even(3) == False\nassert is_even(0) == True\n",
    },
    {
        "name": "list_sum",
        "prompt": 'def total(xs):\n    """Return the sum of a list of numbers."""\n    return ',
        "suffix": "\n\nassert total([]) == 0\nassert total([1, 2, 3]) == 6\nassert total([5]) == 5\n",
    },
    {
        "name": "is_positive",
        "prompt": 'def is_positive(n):\n    """Return True if n > 0."""\n    return ',
        "suffix": "\n\nassert is_positive(5) == True\nassert is_positive(0) == False\nassert is_positive(-3) == False\n",
    },
    {
        "name": "count_chars",
        "prompt": 'def count(s, ch):\n    """Return how many times ch appears in s."""\n    return ',
        "suffix": "\n\nassert count('hello', 'l') == 2\nassert count('', 'a') == 0\nassert count('abc', 'b') == 1\n",
    },
    {
        "name": "double_it",
        "prompt": 'def double(x):\n    """Return x doubled."""\n    return ',
        "suffix": "\n\nassert double(3) == 6\nassert double(0) == 0\nassert double(-2) == -4\n",
    },

    # ===== 14 new HumanEval-style (Phase 14 S1 expansion) =====
    {
        "name": "max_of_two",
        "prompt": 'def max2(a, b):\n    """Return the larger of two numbers."""\n    return ',
        "suffix": "\n\nassert max2(3, 7) == 7\nassert max2(10, 4) == 10\nassert max2(-1, -5) == -1\n",
    },
    {
        "name": "min_of_two",
        "prompt": 'def min2(a, b):\n    """Return the smaller of two numbers."""\n    return ',
        "suffix": "\n\nassert min2(3, 7) == 3\nassert min2(10, 4) == 4\nassert min2(-1, -5) == -5\n",
    },
    {
        "name": "abs_value",
        "prompt": 'def absv(n):\n    """Return absolute value of n."""\n    return ',
        "suffix": "\n\nassert absv(5) == 5\nassert absv(-7) == 7\nassert absv(0) == 0\n",
    },
    {
        "name": "is_negative",
        "prompt": 'def is_negative(n):\n    """Return True if n < 0."""\n    return ',
        "suffix": "\n\nassert is_negative(-3) == True\nassert is_negative(0) == False\nassert is_negative(5) == False\n",
    },
    {
        "name": "list_length",
        "prompt": 'def length(xs):\n    """Return the number of elements in xs."""\n    return ',
        "suffix": "\n\nassert length([]) == 0\nassert length([1, 2, 3]) == 3\nassert length([\"a\"]) == 1\n",
    },
    {
        "name": "first_elem",
        "prompt": 'def first(xs):\n    """Return the first element of xs."""\n    return ',
        "suffix": "\n\nassert first([1, 2, 3]) == 1\nassert first([\"a\", \"b\"]) == \"a\"\nassert first([42]) == 42\n",
    },
    {
        "name": "last_elem",
        "prompt": 'def last(xs):\n    """Return the last element of xs."""\n    return ',
        "suffix": "\n\nassert last([1, 2, 3]) == 3\nassert last([\"a\", \"b\"]) == \"b\"\nassert last([42]) == 42\n",
    },
    {
        "name": "square",
        "prompt": 'def sq(n):\n    """Return n squared."""\n    return ',
        "suffix": "\n\nassert sq(3) == 9\nassert sq(0) == 0\nassert sq(-4) == 16\n",
    },
    {
        "name": "string_upper",
        "prompt": 'def up(s):\n    """Return s in uppercase."""\n    return ',
        "suffix": "\n\nassert up('hello') == 'HELLO'\nassert up('ABC') == 'ABC'\nassert up('') == ''\n",
    },
    {
        "name": "string_lower",
        "prompt": 'def lo(s):\n    """Return s in lowercase."""\n    return ',
        "suffix": "\n\nassert lo('HELLO') == 'hello'\nassert lo('abc') == 'abc'\nassert lo('') == ''\n",
    },
    {
        "name": "is_zero",
        "prompt": 'def is_zero(n):\n    """Return True if n is zero."""\n    return ',
        "suffix": "\n\nassert is_zero(0) == True\nassert is_zero(1) == False\nassert is_zero(-1) == False\n",
    },
    {
        "name": "list_reverse",
        "prompt": 'def rev(xs):\n    """Return xs reversed."""\n    return ',
        "suffix": "\n\nassert rev([1, 2, 3]) == [3, 2, 1]\nassert rev([]) == []\nassert rev([\"a\"]) == [\"a\"]\n",
    },
    # === Cold-start candidates (single-line solution exists but model
    # may not find it without curriculum)
    {
        "name": "fizz_string",
        "prompt": 'def fz(n):\n    """Return \'fizz\' if n%3==0, else str(n)."""\n    return ',
        "suffix": "\n\nassert fz(3) == 'fizz'\nassert fz(6) == 'fizz'\nassert fz(7) == '7'\n",
    },
    {
        "name": "list_max",
        "prompt": 'def lmax(xs):\n    """Return the largest element of xs."""\n    return ',
        "suffix": "\n\nassert lmax([1, 7, 3]) == 7\nassert lmax([-5, -2, -10]) == -2\nassert lmax([42]) == 42\n",
    },
]
