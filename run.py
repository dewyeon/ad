import subprocess
import argparse
parser=argparse.ArgumentParser(description="run")
parser.add_argument("--n", type=int)
args = parser.parse_args()
n = args.n
root = 'results'

if args.n == 0:
    for cl in ['bottle', 'cable', 'leather']:
        log_name = cl
        subprocess.call(f"python main.py --dataset mvtec --phase test --project_root_path {root}  --class_name {cl} --gpu {n} > logs/{log_name}.txt", shell=True)

if args.n == 1:
    for cl in ['wood', 'capsule', 'carpet', 'metal_nut']:
        log_name = cl
        subprocess.call(f"python main.py --dataset mvtec --phase test --project_root_path {root}  --class_name {cl} --gpu {n} > logs/{log_name}.txt", shell=True)

if args.n == 2:
    for cl in ['grid', 'hazelnut','zipper', 'transistor']:
        log_name = cl
        subprocess.call(f"python main.py --dataset mvtec --phase test --project_root_path {root} --class_name {cl} --gpu {n} > logs/{log_name}.txt", shell=True)

if args.n == 3:
    for cl in ['pill', 'screw', 'tile', 'toothbrush']:
        log_name = cl
        subprocess.call(f"python main.py --dataset mvtec --phase test --project_root_path {root}  --class_name {cl} --gpu {n} > logs/{log_name}.txt", shell=True)

if args.n == 4:
    for cl in [0,1,2]:
        for load in [32]:
            for inp in [32]:
                log_name = str(cl) + str(inp) + str(load)
                subprocess.call(f"python main.py --dataset cifar10 --phase test --load_size {load} --input_size {inp} --project_root_path {root}  --label {cl} --gpu {n} > logs/{log_name}.txt", shell=True)

if args.n == 5:
    for cl in [3,4]:
        for load in [32]:
            for inp in [32]:
                log_name = str(cl) + str(inp) + str(load)
                subprocess.call(f"python main.py --dataset cifar10 --phase test --load_size {load} --input_size {inp} --project_root_path {root}  --label {cl} --gpu {n} > logs/{log_name}.txt", shell=True)

if args.n == 6:
    for cl in [5,6,7]:
        for load in [32]:
            for inp in [32]:
                log_name = str(cl) + str(inp) + str(load)
                subprocess.call(f"python main.py --dataset cifar10 --phase test --load_size {load} --input_size {inp} --project_root_path {root}  --label {cl} --gpu {n} > logs/{log_name}.txt", shell=True)

if args.n == 7:
    for cl in [8,9]:
        for load in [32]:
            for inp in [32]:
                log_name = str(cl) + str(inp) + str(load)
                subprocess.call(f"python main.py --dataset cifar10 --phase test --load_size {load} --input_size {inp} --project_root_path {root}  --label {cl} --gpu {n} > logs/{log_name}.txt", shell=True)
