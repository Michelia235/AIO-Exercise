import math

def calc_precision(tp:int, fp: int, fn: int) -> float:
    return tp / (tp + fp)

def calc_recall(tp: int, fp: int, fn: int) -> float:
    return tp / (tp + fn)


def calc_f1_score(tp: int, fp: int, fn: int) -> float:
    #check tf, fp, fn must be integers
    if not (isinstance(tp, int) and isinstance(fp, int) and isinstance(fn, int)):
        print("All inputs must be integers")
        return None
    #check tp, fp, fn must be positive
    if tp <= 0 or fp <= 0 or fn <= 0:
        print("All inputs must be positive")
        return None

    precision = calc_precision(tp, fp, fn)
    recall = calc_recall(tp, fp, fn)
  
    if precision + recall == 0:
        f1_score = 0.0 
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    print("Persicion is: ", precision)
    print("Recall is: ",recall)
    print("F1_score is: ",f1_score)
    return f1_score

assert round(calc_f1_score(tp=2, fp=3, fn=5), 2) == 0.33
#Câu 1:
print(round(calc_f1_score(tp=2, fp=4, fn=5), 2))

#Các test case khác theo đề bài
calc_f1_score(tp='a', fp=2, fn=4)
calc_f1_score(tp=1.5, fp=5, fn=4)
calc_f1_score(tp=-5, fp=5, fn=4)