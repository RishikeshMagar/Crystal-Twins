from matbench.bench import MatbenchBenchmark
from pymatgen.core.structure import Structure


mb = MatbenchBenchmark(autoload=False,subset=[
        "matbench_log_gvrh"])

for task in mb.tasks:
    task.load()
    for fold in task.folds:
        print(fold)
        train_df = task.get_train_and_val_data(fold, as_type="df")
        structure, target = train_df.iloc[0]
        #crystal = str_files[0]
        print(train_df.iloc[0].name)
        print(structure)
        print(target)


        break
        #print(train_outputs)