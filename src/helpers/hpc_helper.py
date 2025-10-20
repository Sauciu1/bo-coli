import pandas as pd



def hit_stats(hit_df, hyper_param_cols=["noise", "tech_repeats"]):
    output_dict = {}
    for name, group in hit_df.groupby(hyper_param_cols):

        wd = {
            "TP_index": group.loc[group["assumed_hit"] & group["true_hit"], "trial_index"].to_list(),
            "FP_index": group.loc[group["assumed_hit"] & ~group["true_hit"], "trial_index"].to_list(),
            "TN_index": group.loc[~group["assumed_hit"] & ~group["true_hit"], "trial_index"].to_list(),
            "FN_index": group.loc[~group["assumed_hit"] & group["true_hit"], "trial_index"].to_list(),
        }
        wd = wd | {
            "TP": len(wd["TP_index"]),
            "FP": len(wd["FP_index"]),
            "TN": len(wd["TN_index"]),
            "FN": len(wd["FN_index"]),
        }
        total = wd["TP"] + wd["FP"] + wd["TN"] + wd["FN"]
        def sdiv(num, den):
            return num / den if den != 0 else 0.0
        wd = wd | {
            "Precision": sdiv(wd["TP"], wd["TP"] + wd["FP"]),
            "Recall": sdiv(wd["TP"], wd["TP"] + wd["FN"]),
            "Specificity": sdiv(wd["TN"], wd["TN"] + wd["FP"]),
            "Accuracy": sdiv(wd["TP"] + wd["TN"], total),
        }
        wd = wd | {
            "F1": sdiv(2 * wd["Precision"] * wd["Recall"], wd["Precision"] + wd["Recall"]),
        }
        # Use the group name (tuple of hyperparameter values) as key
        output_dict[name] = wd

    output_df = pd.DataFrame.from_dict(output_dict, orient='index')
    # If hyper_param_cols > 1, expand tuple index to columns
    if len(hyper_param_cols) > 1:
        output_df.index = pd.MultiIndex.from_tuples(output_df.index, names=hyper_param_cols)
        output_df.reset_index(inplace=True)
    else:
        output_df.index.name = hyper_param_cols[0]
        output_df.reset_index(inplace=True)
    return output_df