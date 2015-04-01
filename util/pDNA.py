def complement(strV):
    dictDNA = {'A':'T',
               'T':'A',
               'G':'C',
               'C':'G'}
    return ''.join([ dictDNA[base] for base in strV ])

def revComp(strV):
    return complement(strV[::-1])
