"""
Here the function to generate new feature from the time evolution of a series as gathered.
"""

import pandas as pd


def _norm_1_to_0(s):
    """
    Normalize such that 1 corresponds to score 0
    """
    return (s - 1).clip(lower=0)


# small_step functions are needed to estimate the value-time-step size to use as a threshold in function below
# Two versions are offered

def small_step_from_steps(s, quantile=0.1):
    """
    Look at actual absolute steps and pick a certain small quantile
    """
    return s.diff().abs().quantile(quantile)


def small_step_from_range(s, fraction= 1/1000, quantile=0.01):
    """
    Estimate a step size from the total range of the data
    """
    return (s.quantile(1 - quantile) - s.quantile(quantile)) * fraction


default_small_step = small_step_from_steps
                          


def one_min_max_agg(s, *, min_length=True, clip_steps=None):
    """
    0 if series goes from start, monotonously to min/max, then to max/min and then to end
    >0 otherwise
    """
    s_small_step = default_small_step(s)

    if min_length is True:
        min_length = s_small_step

    if clip_steps is True:
        clip_steps = s_small_step

    start = s.iloc[0]
    end = s.iloc[-1]
    min_val = s.min()
    max_val = s.max()
    max_first = 1 if s.idxmax() < s.idxmin() else -1

    direct_length = 2 * (max_val - min_val) + max_first * (end - start)

    if min_length is not None:
        direct_length = max(
            direct_length, min_length
        )  # "regularized" with min_length since otherwise for start=end there are artifically large scores

    steps = s.diff().abs()

    if clip_steps is not None:
        steps = steps.mask(steps < min_length, 0)

    series_length = steps.sum()
    
    length_ratio = series_length / direct_length
    
    length_ratio -= 1
    if length_ratio < 0:
        length_ratio = 0

    return length_ratio


def fluct(s, window, *, name_format="fluct_{}", clip_steps=True, standardize=True):
    """
    0 when time evolution is a straight line
    >0 otherwise
    
    Delta x steps are effectively assumed to be 1, so it is advisable to normalize y values to a similar range
    
    May have incorrect results for NaNs
    """
    s_small_step = default_small_step(s)

    if clip_steps is True:
        clip_steps = s_small_step

    if standardize:
        s = s / s.diff().abs().mean() * window

    s_steps_sqr = s.diff().pow(2)

    if clip_steps is not None:
        s_steps_sqr = s_steps_sqr.mask(s_steps_sqr <= clip_steps ** 2, 0)

    s_tot_length = (
        (s_steps_sqr + 1).pow(0.5).fillna(0).cumsum()
    )  # precalculate cumulated sum of segment lengths

    straight_length = (window ** 2 + s.diff(window).pow(2)).pow(0.5)
    
    length_ratio = s_tot_length.diff(window) / straight_length

    return _norm_1_to_0(length_ratio).fillna(0).rename(
        name_format.format(s.name)
    )


def one_hump_agg(s, *, min_length=None, clip_steps=None, norm=True):
    """
    For norm=True:
    =0 for a monotonic development and a single hump or dip
    >0 otherwise
    
    For norm=False:
    =0 if the series is monotonous
    >0 otherwise
    
    use it with .agg() rather than .apply() since it needs a Series as input
    """
    assert not (norm is False and min_length), "Setting min_length for norm=False does not make sense"
    
    start = s.iloc[0]
    end = s.iloc[-1]

    steps = s.diff().abs()

    if clip_steps is not None:
        steps = steps.mask(steps <= clip_steps, 0)

    length = steps.sum()

    if norm:
        direct_length = 2 * (s.max() - s.min()) - abs(end - start)
        
        if min_length is not None:
            direct_length = max(
                direct_length, min_length
            )  # "regularized" with min_length since otherwise for start=end there are artifically large scores

        result = length / direct_length
        # using (length - abs(end-start)) / (...) would be troublesome since you may divide by small numbers

        result -= 1
        if result < 0:
            result = 0      
    else:
        direct_length = abs(end - start)
        
        result = (length - direct_length) / 2   # / 2 so that it roughly corresponds to peak size

    return result


def one_hump(
    s, window, *, min_length=True, name_format="hump_{}", norm=True, clip_steps=None
):
    """
    For norm=True:
    =0 for a monotonic development and a single hump or dip
    >0 otherwise
    
    For norm=False:
    =0 if the series is monotonous
    >0 otherwise
    
    Optimized for overlapping windows and does own windowing
    """
    assert not (norm is False and min_length), "Setting min_length for norm=False does not make sense"
    
    s_small_step = default_small_step(s)

    if min_length is True:
        min_length = s_small_step

    if clip_steps is True:
        clip_steps = s_small_step

    start = s
    end = s.shift(-window)

    steps = s.diff().abs()

    if clip_steps is not None:
        steps = steps.mask(steps <= clip_steps, 0)

    cumu_abs = steps.cumsum().fillna(0)

    length = cumu_abs.shift(-window) - cumu_abs

    if norm:
        s_max = s.rolling(window).max()
        s_min = s.rolling(window).min()
        direct_length = 2 * (s_max - s_min) - abs(end - start)
        
        if min_length not in (None, False):
            direct_length = direct_length.clip(lower=min_length)
            # "regularized" with min_length since otherwise for start=end there are artifically large scores

        length_ratio = length / direct_length
        # using (length - abs(end-start)) / (...) would be troublesome since you may divide by small numbers
        
        result = _norm_1_to_0(length_ratio)
    else:
        direct_length = abs(end - start)
        
        result = (length - direct_length) /  2   # / 2 so that it roughly corresponds to peak size

    return result.rename(name_format.format(s.name))


def extra_hump(s, window, *, name_format="extrahump_{}", clip_steps=None):
    """
    Used to detect cloud dips
    Basically calls `one_hump` with `norm=False` but centers the window
    """
    return (
        one_hump(
            s,
            window,
            norm=False,
            min_length=None,
            name_format=name_format,
            clip_steps=clip_steps,
        )
        .shift(window // 2)
        .fillna(0)
    )


def spike_values(s, N=1):
    """
    Simple quantifier of Spikes in a series.
    = min(Max - First, Max - Last)
    """
    s_max = s.rolling(2 * N + 1, center=True).max()
    s_first = s.shift(N)
    s_last = s.shift(-N)

    spike_first_last = pd.DataFrame(
        {"first": s_max - s_first, "last": s_max - s_last}
    )

    result = spike_first_last.min(axis=1)
    return result


def saw_tooth(
    s,
    min_jump=0.0015,
    jump_window="3min",
    time_between_jumps="15min",
    num_too_many_in_window=2,
):
    """
    Value > 0 if sizable jump up (but not too frequent) followed by a small dip
    
    Calculate size of saw tooth jumps.
    Quantifies any jumps up but not specifically tuned to a global saw tooth shape. May have spurious results outside of saw-tooth mode.
    May need tuning for a particular feature but already useful to do quick quantitative analysis.    
    """
    result = s.rolling(jump_window).agg(
        lambda x: (x.max() - x.iloc[0]) * (x.max() >= x.iloc[-1])
    )
    # =(max - first) but only if (max > last)

    is_large = result > min_jump

    many_large_recently = (
        is_large.rolling(time_between_jumps).sum().ge(num_too_many_in_window)
    )
    is_small = is_large.eq(0)

    result = result.mask(many_large_recently | is_small, 0)

    return result
