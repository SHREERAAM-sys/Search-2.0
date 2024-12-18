import functools

from torch.profiler import ProfilerActivity, profile, record_function

ROW_LIMIT = 10


def profile_colpali(enable_profiling=True):
    def decorator(func):
        @functools.wraps(func)
        def wrap_function(*args, **kwargs):
            if enable_profiling:
                with profile(
                    activities=[ProfilerActivity.CPU],
                    profile_memory=True,
                    record_shapes=True,
                ) as prof:
                    with record_function("colpali_inference"):
                        result = func(*args, **kwargs)
                print(
                    prof.key_averages().table(
                        sort_by="self_cpu_memory_usage", row_limit=ROW_LIMIT
                    )
                )
                print(
                    prof.key_averages().table(
                        sort_by="cpu_memory_usage", row_limit=ROW_LIMIT
                    )
                )
            else:
                result = func(*args, **kwargs)
            return result

        return wrap_function

    return decorator
