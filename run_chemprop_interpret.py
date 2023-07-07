from chemprop.interpret import interpret
from chemprop.args import InterpretArgs
import sys
import os
import pandas as pd


if __name__ == "__main__":
    test_path = sys.argv[1]
    preds_path = sys.argv[2]
    main_models_path = sys.argv[3]
    property_name = sys.argv[4]

    result_pd = pd.read_csv(test_path)
    property_list = property_name.split(";")

    for prop in property_list:
        arguments = [
            '--data_path', test_path,
            # '--preds_path', preds_path,
            # '--checkpoint_path', os.path.join(main_models_path,'{}.pt'.format(prop)),
            # '--checkpoint_dir', main_models_path,
            '--checkpoint_path', main_models_path,
            '--features_generator',f'{prop}',
            '--no_cuda',
            '--property_id','1',
            '--smiles','smiles_standarized',
            '--num_workers','2'
            ]
        try:
            args = InterpretArgs().parse_args(arguments)
            result = interpret(args=args)
            result_pd[prop] = result['label'].tolist()
            result_pd['{}_rationale'.format(prop)] = result['rationale'].tolist()
            result_pd['{}_score'.format(prop)] = result['rationale_score'].tolist()
        except Exception as exp:
            print("####ERROR:" + str(exp))

        result_pd.to_csv(preds_path, sep=",", header=True, index=False)

# COMMAND
# python interpret.py /var/www/deep_pk/deep_pk/static/examples/test.csv
