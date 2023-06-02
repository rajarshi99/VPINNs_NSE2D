import subprocess
import yaml
import os

# Load the input YAML file
with open("input.yaml", "r") as f:
    input_data = yaml.safe_load(f)

Method_Arrays = ["VarPINNs", "VarPINNs_SUPG"]
epsilons = [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
learning_rates = [5e-2,1e-3,5e-3]
neurons_per_layer = [5,10,20,30]

experiment_count = 0

with open("experiment_status", "w") as f:
    f.write("")


# Modify the input YAML file for each experiment and save it
for method in Method_Arrays:
    for epsilon in epsilons:
        for learning_rate in learning_rates:
            for neurons in neurons_per_layer:
                print(f"Running experiment {experiment_count} with {method} and epsilon = {epsilon} and learning rate = {learning_rate} and neurons = {neurons}")
                # Modify the input_data dictionary for the current experiment
                input_data["model_run_params"]["var_form"] = method
                input_data["model_run_params"]["eps"] = epsilon
                input_data["model_run_params"]["learning_rate"] = learning_rate
                # input_data["model_run_params"]["max_iter"] = 50000
                input_data["model_run_params"]["NN_model"] = [2] + [neurons]*3 + [1]
                input_data["model_run_params"]["save_directory"] = f"Output/SingularlyPerturbed_Experiments/" + str(experiment_count)


                # Save the modified input_data dictionary to a new YAML file
                with open(f"input_{experiment_count}.yaml", "w") as f:
                    yaml.dump(input_data, f)


                # Run the experiment using subprocess
                cmd = f"python3 main/SingularlyPerturbed/hp-VPINN-Singularly_perturbed_2D.py input_{experiment_count}.yaml"
                proc = subprocess.Popen(cmd, shell=True)
                proc.wait()

                # Get the PID of the process
                pid = proc.pid

                # Check if the process exists
                if pid is not None:
                    try:
                        # Terminate the process
                        os.kill(pid, 9)
                    except ProcessLookupError:
                        # Handle the case where the process does not exist
                        pass

                # remove the YAML file using os
                os.remove(f"input_{experiment_count}.yaml")

                with open("experiment_status", "a") as f:
                    f.write(f"{experiment_count}\n")

                experiment_count += 1
                
