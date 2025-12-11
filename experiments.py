from copy import deepcopy
from main2 import run_experiment, CFG
import itertools

# ----------------------------------------------------------
# Перебираемые гиперпараметры
# ----------------------------------------------------------
activations = ["ReLU", "LeakyReLU", "GELU", "ELU", "Mish", "Sigmoid", "Tanh"]
optimizers = ["SGD"] # "Adam", "SGD"
batch_sizes = [32, 64]
img_sizes = [96, 128]
conv_archs = [
    [32, 64],
    [32, 64, 128],
    [64, 128, 256]
]
augment_flags = [True, False]
lrs = [1e-3, 5e-4]

# ----------------------------------------------------------
# Генерация всех комбинаций
# ----------------------------------------------------------
search_space = itertools.product(
    optimizers,
    batch_sizes,
    img_sizes,
    conv_archs,
    augment_flags,
    lrs,
    activations
)


# Ограничим 20 экспериментами
MAX_RUNS = 20
runs = 0

# ----------------------------------------------------------
# Цикл экспериментов
# ----------------------------------------------------------
for opt, bs, img, conv, aug, lr, act in search_space:
    cfg = deepcopy(CFG)
    cfg["optimizer"] = opt
    cfg["batch_size"] = bs
    cfg["img_size"] = img
    cfg["conv_filters"] = conv
    cfg["use_augmentation"] = aug
    cfg["lr"] = lr
    cfg["activation"] = act

    
    print("\n===============================")
    print(" RUNNING EXPERIMENT #", runs + 1)
    print("===============================\n")
    
    run_experiment(cfg)

    runs += 1
    if runs >= MAX_RUNS:
        print("Максимальное число экспериментов достигнуто.")
        break
