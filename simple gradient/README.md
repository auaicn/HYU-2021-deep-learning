## Goal : to find `W` , `b`

## Sequence

1. 학습 데이터 1000 개 생성
	테스트 샘플 100 개 생성
``` python
for i in range(numberTrainingSamlples):
	x1_training.append(random.uniform(-10,10))
	x2_training.append(random.uniform(-10,10))
	if(x1_training[-1] + x2_training[-1] > 0):
		y_training.append(1)
	else:
		y_training.append(0)
```

2. 

### What I Learned
- `random` module is included, whereas `numpy` module has to be installed
  so `random` is differentf rom `numpy.random`
- Forward-Declaration needed for python's 
- python function requires strict type. So Explicit Type Annotation was effective

even though python is somewhat free at handling big-integer,
- underflow occurs when calculating e^(-x) when `x >= 730'
- overflow occurs when calculating e^(x) when `x >= 730'
maybe it's because different integer

- float handling... if a float value become close to 0(or 1) then, it is treated as real 0(or 1)

### To Search
[] except: block's second line code `exit()` or `quit()` was not executed -> why?