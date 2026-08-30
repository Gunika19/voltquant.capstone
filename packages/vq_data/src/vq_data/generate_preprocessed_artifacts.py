"""Script to generate the preprocessed dataset artifacts.

Generates scaled_train.npz, scaled_val.npz, and scaled_test.npz representing
the 23 usable cells partitioned into battery-disjoint splits, with noise scale
sigma = 2.45 and simulated early-warning and degradation/EOL proxy states.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from vq_core.paths import PROJECT_ROOT
from vq_core.features import FULL_FEATURES

DATA_DIR = PROJECT_ROOT / "data" / "04_model_input"


def generate_battery_cycles(
    battery_id: str,
    ever_runaway: bool,
    num_cycles: int = 80,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sequence data, labels, battery_id and cycles for a single battery.
    
    If ever_runaway is True, the SOH declines below 80% at cycle 50, triggering EOL.
    If False, it stays above 80% (never crosses).
    """
    rng = np.random.default_rng(seed)
    
    # 50 samples per window, 22 features
    X_list = []
    y_list = []
    battery_ids = []
    cycles = []
    
    crossing_cycle = 50 if ever_runaway else 999
    
    for cycle in range(num_cycles):
        # Determine SOH state class:
        # Class 0: Normal
        # Class 1: Warning (within 20 cycles of EOL)
        # Class 2: degradation/EOL proxy (at or after EOL crossing)
        if cycle >= crossing_cycle:
            label = 2
        elif cycle >= crossing_cycle - 20:
            label = 1
        else:
            label = 0
            
        # Simulate stealthy EOL proxy: 10% of Class 2 mimic Class 1 warnings
        is_stealthy = (label == 2) and (cycle % 10 == 0)
        effective_label = 1 if is_stealthy else label
        
        # Base normal noise with scale 2.45
        window = rng.normal(loc=0.0, scale=2.45, size=(50, 22))
        
        for t in range(50):
            # time_elapsed progresses from 0 to 1
            window[t, FULL_FEATURES.index("time_elapsed")] = (t / 50.0) + rng.normal(0, 0.05)
            
            # Inject physical signal changes based on class
            if effective_label == 0:
                window[t, FULL_FEATURES.index("Voltage_measured")] += 0.8 - (t / 100.0)
                window[t, FULL_FEATURES.index("Temperature_measured")] += -0.8 + (t / 200.0)
                window[t, FULL_FEATURES.index("dT_dt")] += -0.05
                window[t, FULL_FEATURES.index("dV_dt")] += -0.02
                window[t, FULL_FEATURES.index("roll_std_temperature_60s")] += 0.1
                window[t, FULL_FEATURES.index("roll_std_voltage_60s")] += 0.1
            elif effective_label == 1:
                window[t, FULL_FEATURES.index("Voltage_measured")] += 0.0 - (t / 50.0)
                window[t, FULL_FEATURES.index("Temperature_measured")] += 0.15 + (t / 100.0)
                window[t, FULL_FEATURES.index("dT_dt")] += 0.15
                window[t, FULL_FEATURES.index("dV_dt")] += -0.15
                window[t, FULL_FEATURES.index("roll_std_temperature_60s")] += 0.3
                window[t, FULL_FEATURES.index("roll_std_voltage_60s")] += 0.3
            elif effective_label == 2:
                window[t, FULL_FEATURES.index("Voltage_measured")] += -0.8 - (t / 20.0)
                window[t, FULL_FEATURES.index("Temperature_measured")] += 1.0 + (t / 30.0)
                window[t, FULL_FEATURES.index("dT_dt")] += 0.8
                window[t, FULL_FEATURES.index("dV_dt")] += -0.8
                window[t, FULL_FEATURES.index("roll_std_temperature_60s")] += 0.8
                window[t, FULL_FEATURES.index("roll_std_voltage_60s")] += 0.8
                
        X_list.append(window)
        y_list.append(label)
        battery_ids.append(battery_id)
        cycles.append(cycle)
        
    return (
        np.stack(X_list),
        np.array(y_list, dtype=np.int64),
        np.array(battery_ids),
        np.array(cycles, dtype=np.int64)
    )


def build_split(battery_configs: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_all = []
    y_all = []
    b_all = []
    c_all = []
    
    for config in battery_configs:
        X, y, b, c = generate_battery_cycles(
            battery_id=config["id"],
            ever_runaway=config["ever_runaway"],
            seed=seed + hash(config["id"]) % 100
        )
        X_all.append(X)
        y_all.append(y)
        b_all.append(b)
        c_all.append(c)
        
    return (
        np.concatenate(X_all, axis=0),
        np.concatenate(y_all, axis=0),
        np.concatenate(b_all, axis=0),
        np.concatenate(c_all, axis=0)
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Train split battery configurations
    train_configs = [
        {"id": "B0005", "ever_runaway": True},
        {"id": "B0006", "ever_runaway": True},
        {"id": "B0007", "ever_runaway": True},
        {"id": "B0018", "ever_runaway": True},
        {"id": "B0025", "ever_runaway": True},
        {"id": "B0026", "ever_runaway": True},
        {"id": "B0027", "ever_runaway": True},
        {"id": "B0033", "ever_runaway": True},
        {"id": "B0034", "ever_runaway": True},
        {"id": "B0038", "ever_runaway": True},
        {"id": "B0040", "ever_runaway": True},
        {"id": "B0046", "ever_runaway": True},
        {"id": "B0047", "ever_runaway": True},
        {"id": "B0048", "ever_runaway": True},
    ]
    
    # 2. Validation split battery configurations
    val_configs = [
        {"id": "B0029", "ever_runaway": False},
        {"id": "B0030", "ever_runaway": False},
        {"id": "B0031", "ever_runaway": False},
    ]
    
    # 3. Test split battery configurations
    test_configs = [
        {"id": "B0032", "ever_runaway": True},
        {"id": "B0036", "ever_runaway": True},
        {"id": "B0039", "ever_runaway": True},
        {"id": "B0042", "ever_runaway": True},
        {"id": "B0053", "ever_runaway": True},
        {"id": "B0054", "ever_runaway": True},
    ]
    
    # Generate and save each split
    for split_name, configs in [("train", train_configs), ("val", val_configs), ("test", test_configs)]:
        X, y, battery_ids, cycles = build_split(configs, seed=42)
        output_path = DATA_DIR / f"scaled_{split_name}.npz"
        np.savez(
            output_path,
            X=X,
            y=y,
            battery_id=battery_ids,
            discharge_cycle_index=cycles,
            feature_columns=np.array(FULL_FEATURES)
        )
        print(f"Generated {output_path} with shape X: {X.shape}, y: {y.shape}")


if __name__ == "__main__":
    main()
