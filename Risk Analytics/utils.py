import json

import config

with open(config.factor_codes) as file:
    factor_codes = json.load(file)


def cat_decoder(sex, housing, savings_acc, checking_acc, purpose, job):
    """
    Convert the strings to int for the model input
    :return: decoded factors
    """
    sex = factor_codes['sex'][sex]
    housing = factor_codes['Housing'][housing]
    savings_acc = factor_codes['Saving_accounts'][savings_acc]
    checking_acc = factor_codes['Checking_account'][checking_acc]
    purpose = factor_codes['Purpose'][purpose]
    job = factor_codes['job'][job]
    return sex, housing, savings_acc, checking_acc, purpose, job


def translator(prediction, prob=False):
    """
    Converts the prediction to the user readable form
    :param prediction: prediction from the model
    :param prob: Prob scores of the prediction
    :return: converted prediction
    """
    if prob:
        out = dict()
        out["Bad"] = prediction[0][0]
        out["Good"] = prediction[0][1]
        return out
    else:
        return factor_codes['Risk'][prediction]


def factors_organiser(age, sex, job, housing, saving_accounts, checking_account,
                      credit_amount, duration, purpose):
    """
    Prepares and organises the input factors for the model prediction
    :return: factors in 2d Array
    """
    # Fixme: Below is not the right way to do this
    sex, housing, saving_accounts, checking_account, purpose, job = cat_decoder(sex, housing, saving_accounts,
                                                                                checking_account, purpose, job)

    factors = [age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose]

    return [factors]  # For converitng 2d array
