import subprocess

if __name__ == '__main__':
    lrs = [3e-5, 3e-4, 3e-3, 3e-1]
    # lrs = [3e-4, 3e-2]
    powers = [0.5, 0.25, 0, -0.01, -0.1]
    for lr in lrs:
        for power in powers:
            filename = f"notes/results_lr{lr:.0e}_power{power}.txt"
            command = f"python train.py --data cifar10 --model lenet --batch-size 64 --num-epoch 20 --print-step 300 " \
                      f"--lr {lr} --power {power}"
            with open(filename, "w") as file:
                subprocess.run(command, stdout=file, shell=True)
