from itertools import combinations 





def Prune_func(higher_lvl,curr_lvl,l):
    pruned = [] 
    for item in curr_lvl: 
        combos = generate_combos(item,l)
        higher_lvl,flag = list(higher_lvl),0 
        for ele in combox: 
            if tuple(sorted(ele)) in highe