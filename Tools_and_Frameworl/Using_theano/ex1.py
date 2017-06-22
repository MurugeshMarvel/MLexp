from theano import tensor as T
from theano import function
from theano import pp

#declare variable

a = T.scalar()
b = T.scalar()

res = (a**2) + (b**2) + (2*a*b)
func = function([a,b],res)
choice = 'y'
while choice == 'y':
	x = input("Enter the x")
	y = input("Enter the y")
	print (func(x,y))
	choice = raw_input("Enter y to continue")
	choice = choice.lower()



