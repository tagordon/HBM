from distributions import *
from exo_archive import get_query_results

def get_params(name, author=None, year=None, prefixes=['st', 'pl', 'sy']):
    
    params = get_query_results(name, author=author, year=year)
    
    param_dicts = []
    refs = []
    
    all_param_dicts = {}
    for param_set in params:
        
        all_keys = np.unique(np.array([k.split('err')[0] for k in param_set.keys()]))
        keys = all_keys[[True if np.any([p in k[:2] for p in prefixes]) else False for k in all_keys]]
        
        ref = param_set['pl_refname'].split('refstr=')[1].split(' href')[0]
        
        pd = {}
        for k in keys:

            try:
                if isinstance(param_set[k], (int, float)) and not isinstance(param_set[k], bool):
                    pd[k] = (param_set[k], param_set[k + 'err1'], param_set[k + 'err2'])
                if isinstance(param_set[k], bool):
                    pd[k] = (param_set[k], np.nan, np.nan)
            except KeyError:
                try:
                    pd[k] = (param_set[k], np.nan, np.nan)
                except:
                    pass
                
        all_param_dicts[ref] = pd
            
    return all_param_dicts

def get_priors(name, author=None, year=None, allow_assymetric=True, prefixes=['st', 'pl', 'sy']):
    
    trunc_at_zero = ['pl_bmasse', 'pl_bmassj', 'pl_dens', 'pl_masse', 'pl_massj', 'pl_orbsmax', 'pl_ratdor']
    all_param_dicts = get_params(name, author=author, year=year, prefixes=prefixes)
    
    all_priors_dicts = {}
    for ref, pd in all_param_dicts.items():
        priors_dict = {}
        for k, p in pd.items():
            
            low = 0.0
            high = np.inf
            if k == 'pl_orbeccen':
                high = 1.0

            p = np.array(p, dtype=float)
            if allow_assymetric:
                if np.all(np.isfinite(p)):
                    if k in trunc_at_zero:
                        priors_dict[k] = trunc_assymetric_normal_prior(*p, low, high)
                    else:
                        priors_dict[k] = assymetric_normal_prior(*p)
                elif np.any(np.isfinite(p[1:])):
                    if k in trunc_at_zero:
                        priors_dict[k] = trunc_normal_prior(p[0], p[1:][np.isfinite(p[1:])][0], low, high)
                    else:
                        priors_dict[k] = normal_prior(p[0], p[1:][np.isfinite(p[1:])][0])
                else:
                    priors_dict[k] = p[0]
            else:
                if np.any(np.isfinite(p[1:])):
                    if k in trunc_at_zero:
                        priors_dict[k] = trunc_normal_prior(p[0], np.nanmax(p[1:]), low, high)
                    else:
                        priors_dict[k] = normal_prior(p[0], np.nanmax(p[1:]))
                else:
                    priors_dict[k] = p[0]
                    
        all_priors_dicts[ref] = priors_dict
        
    return all_priors_dicts