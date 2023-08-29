import yaml
import sys

if len(sys.argv) < 2:
    print('Error: need to specify the input.yaml file')
    exit(0)

# read the input.yaml file
with open(sys.argv[1], 'r') as f:
    input_data = yaml.safe_load(f)

counter = 0
quad_points = [80, 40, 20, 10, 5]

for qp in quad_points:
    input_data['model_run_params']['num_quad_points'] = qp
    input_data['model_run_params']['num_elements_x'] = int(160/qp)
    input_data['model_run_params']['num_elements_y'] = int(160/qp)

    counter += 1
    input_data['experimental_params']['output_folder'] = f'Output/NSE2D/exp_curri_h_exp_smaller_net{counter:02}'
    input_data['mlflow_parameters']['mlflow_run_prefix'] = f'exp_curri_h_exp_smaller_net{counter:02}'
    with open(f'run{counter:02}.yaml', 'w') as outf:
        yaml.dump(input_data, outf)
    print(f"qp = {qp}, counter = {counter}")
