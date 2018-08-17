import cProfile
import functools
import pickle
import os
from accelerate import profiler
import datetime
from foolbox.data_mgmt import set_credentials as set_cred
path_to_data = set_cred.set_path("research_data/fx_and_events/")



def cprofile_analysis(**option_kwargs):
    def outerwrapper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            p = cProfile.Profile()
            try:
                p.enable()
                ret = func(*args, **kwargs)
                p.disable()
                return ret
            finally:
                _refor_time = datetime.datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
                filename = os.path.join(
                    path_to_data + func.__name__
                    +'_' + _refor_time +'_profiler_test.prof')
                p.dump_stats(filename)
                p.print_stats()

                if option_kwargs.get('activate'):
                    os.system("snakeviz " + filename)


        return wrapper
    return outerwrapper


# def accelerate_profile(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         p = profiler.Profile()
#         try:
#             p.enable()
#             ret = func(*args, **kwargs)
#             p.disable()
#             return ret
#         finally:
#             # filename = os.path.expanduser(
#             #     os.path.join('~', func.__name__  '.pstat')
#             # )
#             # profiler.dump_stats(filename)
#
#             p.print_stats()
#
#             _refor_time = datetime.datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
#             fh = open(os.path.join('Z:\Projekte\GTAA\RootFolder\output_data\\' + func.__name__ +'_' + _refor_time +'_profiler_test.pkl'), 'wb')
#             pickle.dump(p, fh)
#             fh.close()
#     return wrapper

# def load_profile_data():

    # import pickle
    # import os
    # from accelerate import profiler
    # fh = open(os.path.join('Z:\Projekte\GTAA\RootFolder\output_data\\retrieve_data_wrapper_20170213_18h11m06s_profiler_test.pkl'), 'rb')
    # profile_data = pickle.load(fh)
    # profiler.plot(profile_data)

    # pass