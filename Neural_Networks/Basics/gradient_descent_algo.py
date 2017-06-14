#GRADIENT DESCENT

while True:
	loss = f(params)
	d_loss_wrt_params = .....#compute gradient
	params -= learning_rate * d_loss_wrt_params
	if <stopping condition is met>:
		return params
