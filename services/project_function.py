from services.strategies import *
from services.optimization_layers import *


def project_function(periodReturns, periodFactRet):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    Strategy = OLS_MVO()
    x = Strategy.execute_strategy(periodReturns, periodFactRet)

    # End to end learning strategy
    # IMPORTANT: since a new object is created each time the project function is called it
    # is always the object's first time making an allocation and as such the e2e net will
    # retrain delta  . . . This will be fixed soon

    # e2e_net = e2e_regularization_estimator(nr_assets = periodReturns.shape[1], estimation_frequency = 1, investPeriod = 1,
    #                                        NumObs= 36,
    #                                        opt_layer= "penalty_regularized_MVO",
    #                                        lr = 0.0001, epochs = 30 , store=True)
    #
    # Strategy = e2e_strategy(e2e_layer=e2e_net, NumObsTraining = 60)
    # x = Strategy.execute_strategy(periodReturns, periodFactRet)

    return x
