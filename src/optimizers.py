import torch
from torch.optim import Optimizer

def get_optimizer(algorithm: str, model: torch.nn.Module) -> Optimizer:
	"""
		algorithm: Algrotihm to return the optimizer from
		model: A module to which we attach the gradient
		ret: Optimizer
	"""

	if algorithm == 'rmsprop':
		optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, eps=1e-06) # TODO: rho=0.9 in original model
	elif algorithm == 'sgd':
		optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False)
	elif algorithm == 'adagrad':
		optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01) # TODO: epsilon=1e-06 in original model
	elif algorithm == 'adadelta':
		optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-06)
	elif algorithm == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
	elif algorithm == 'adamax':
		optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), epsilon=1e-08)
	else:
		raise NotImplementedError("{} not a supported optimizer algorithm".format(algorithm))
	
	return optimizer
