import requests
import json

def get_archive_name(name):
    alias_query = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Lookup/nph-aliaslookup.py?objname=' + name
    res = json.loads(requests.get(alias_query).content)
    status = res['manifest']['lookup_status']

    if status == 'OK':

        archive_name = res['manifest']['resolved_name']
        return archive_name

    else:
        raise Error('planet not found')
        
def get_from_ps(name):
    
    archive_name = get_archive_name(name)
    query = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps+where+pl_name+=+' + '\'' + archive_name + '\'' + '&format=json'
    results = json.loads(requests.get(query).content)
    return results

def get_from_pscomppars(name):
    
    archive_name = get_archive_name(name)
    query = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars+where+pl_name+=+' + '\'' + archive_name + '\'' + '&format=json'
    results = json.loads(requests.get(query).content)
    return results

def get_query_results(name, author=None, year=None, composite=False):
    
    if composite:
        results = get_from_pscomppars(name)
        return results
    
    if (author is None) & (year is None):
        results = get_from_ps(name)
        return results
    
    elif year is None:
        results = get_from_ps(name)
        
        author_str = author.upper()
        
        res = []
        for r in results:
            if (author_str in r['pl_refname']):
                res.append(r)
                
        return res
    
    elif author is None:
        results = get_from_ps(name)
        
        res = []
        for r in results:
            if (year in r['pl_refname']):
                res.append(r)
                
        return res
    
    else:
        results = get_from_ps(name)
    
        author_str = author.upper()

        res = []
        for r in results:
            if (author_str in r['pl_refname']) & (year in r['pl_refname']):
                res.append(r)

        return res