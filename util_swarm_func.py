import time
import functools
import multiprocessing


def run_simulation(simulation, args, instances, processes=1, batch=100, silent=False):
    pool = multiprocessing.Pool(processes=processes)
    timeseries_data_all = []
    edge_data_all = []
    time_data_all = []

    remaining_instances = instances

    prev_time = time.time()
    while remaining_instances > 0:
        n = min(remaining_instances, batch)
        func = functools.partial(simulation, args)
        data_pool = pool.map(func, range(n))

        timeseries_pool, edge_pool, time_pool = zip(*data_pool)

        remaining_instances -= n
        if not silent:
            print('Simulation {}/{}... {:.1f}s'.format(instances - remaining_instances,
                                                       instances, time.time()-prev_time))
        prev_time = time.time()

        timeseries_data_all.extend(timeseries_pool)
        edge_data_all.extend(edge_pool)
        time_data_all.extend(time_pool)

    return timeseries_data_all, edge_data_all, time_data_all
