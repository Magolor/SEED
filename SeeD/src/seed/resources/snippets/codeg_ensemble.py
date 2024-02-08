from seed import *

<<import_snippets>>

SNIPPETS = <<list_of_snippets>>

<<api.api_def(with_kwargs=True)>>
    results = []
    for snippet in SNIPPETS:
        try:
            result = snippet(<<api.asgs>>)
        except Exception as e:
            print(e)
            result = None
        results.append(result)
        
    non_abs_results = [r for r in results if r is not None]
    if len(non_abs_results) == 0:
        return None
    
    # Majority Voting
    counter = Counter(non_abs_results)
    return counter.most_common(1)[0][0]