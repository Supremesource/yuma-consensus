import matplotlib.pyplot as plt
import json

# == MAIN YUMA HERE

def weighted_median_col_sparse(
    stake: list[float],
    score: list[list[tuple[int, float]]],
    columns: int,
    majority: float,
) -> list[float]:

    rows = len(stake)
    zero = 0.0
    use_stake = [s for s in stake if s > zero]
    inplace_normalize(use_stake)
    stake_sum = sum(use_stake)
    stake_idx = list(range(len(use_stake)))
    minority = stake_sum - majority
    use_score = [[zero] * len(use_stake) for _ in range(columns)]
    median = [zero] * columns

    k = 0
    for r in range(rows):
        if stake[r] <= zero:
            continue
        for c, val in score[r]:
            use_score[c][k] = val
        k += 1

    for c in range(columns):
        median[c] = weighted_median(
            use_stake,
            use_score[c],
            stake_idx,
            minority,
            zero,
            stake_sum,
        )

    return median

def weighted_median(stake: list, score: list, partition_idx: list,
                    minority: float, partition_lo: float, partition_hi: float) -> float:
    n = len(partition_idx)
    if n == 0:
        return 0.0
    if n == 1:
        return score[partition_idx[0]]
    assert len(stake) == len(score)
    mid_idx = n // 2
    pivot = score[partition_idx[mid_idx]]
    lo_stake = 0.0
    hi_stake = 0.0
    lower = []
    upper = []
    for idx in partition_idx:
        if score[idx] == pivot:
            continue
        if score[idx] < pivot:
            lo_stake += stake[idx]
            lower.append(idx)
        else:
            hi_stake += stake[idx]
            upper.append(idx)
    if partition_lo + lo_stake <= minority < partition_hi - hi_stake:
        return pivot
    elif minority < partition_lo + lo_stake and len(lower) > 0:
        return weighted_median(stake, score, lower, minority, partition_lo, partition_lo + lo_stake)
    elif partition_hi - hi_stake <= minority and len(upper) > 0:
        return weighted_median(stake, score, upper, minority, partition_hi - hi_stake, partition_hi)
    return pivot

def col_clip_sparse(
    sparse_matrix: list[list[tuple[int, float]]],
    col_threshold: list[float],
) -> list[list[tuple[int, float]]]:
    result: list[list[tuple[int, float]]] = [[] for _ in range(len(sparse_matrix))]
    for i, sparse_row in enumerate(sparse_matrix):
        for j, value in sparse_row:
            if col_threshold[j] < value:
                if 0 < col_threshold[j]:
                    result[i].append((j, col_threshold[j]))
            else:
                result[i].append((j, value))

    return result

# == Helper functions ==

def inplace_col_clip(x: list, col_threshold: list) -> None:
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = min(x[i][j], col_threshold[j])


def inplace_normalize(x: list[int]) -> None:
    x_sum = sum(x)
    for i in range(len(x)):
        x[i] /= x_sum
    return x


def row_sum(x: list) -> list:
    if len(x) == 0:
        return []
    rows = len(x)
    result = [0.0] * rows
    for i in range(rows):
        result[i] = sum(x[i])
    return result

def get_weights_sparse(weights: dict[int, list[tuple[int, int]]]):
    weights_sparse = [[] for _ in range(len(weights))]
    for uid_i, weights_i in weights.items():
        for uid_j, weight_ij in weights_i:
            weights_sparse[uid_i].append((uid_j, float(weight_ij)))
    return weights_sparse


def inplace_row_normalize_sparse(sparse_matrix):
    for sparse_row in sparse_matrix:
        row_sum = sum(value for _, value in sparse_row)
        if row_sum > 0.0:
            sparse_row[:] = [(uid_j, value / row_sum) for uid_j, value in sparse_row]
    return sparse_matrix

# == Utility functions, not part of the main algorithm ==

def plot_consensus(E: float, weight: list[list[tuple[int, float]]], stake_vec: list[float]):
    """
    plots the distribution
    """

    vali_stake_proportion = 1 - E

    miner_rewards = {}
    total_weight = []
    vali_stake = []

    for x in weight:
        weight_sum = 0
        for u, w in x:
            miner_rewards[u] = miner_rewards.get(u, 0) + w
            weight_sum += w
        total_weight.append(weight_sum)

    for weight, stake in zip(total_weight, stake_vec):
        vali_stake.append(weight * (stake * vali_stake_proportion))

    total_vali_stake = sum(vali_stake)
    total_miner_rewards = sum(miner_rewards.values())

    miner_rewards_stake = {k: v / total_miner_rewards * total_vali_stake for k, v in miner_rewards.items()}
    vali_rewards_stake = {i: v for i, v in enumerate(vali_stake) if v > 0}

    # Plotting the histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot miner_rewards_stake
    ax1.bar(miner_rewards_stake.keys(), miner_rewards_stake.values())
    ax1.set_xlabel('Miner')
    ax1.set_ylabel('Reward Stake')
    ax1.set_title('Miner Rewards Stake Distribution')

    # Plot vali_rewards_stake
    ax2.bar(vali_rewards_stake.keys(), vali_rewards_stake.values())
    ax2.set_xlabel('Validator')
    ax2.set_ylabel('Reward Stake')
    ax2.set_title('Validator Rewards Stake Distribution')

    plt.tight_layout()
    plt.savefig('assets/consensus_distribution.png')
    plt.close()
    

def validator_settings() -> dict[tuple[int, int], tuple[list[int], list[int]]]:
    """
    The distribution dict represents, (uid, stake), (uids, weights) for each validator

    This acts like a set_weights function
    """

    with open('weights.json', 'r') as f:
        data = json.load(f)

    distribution_dict = {}
    for key, value in data.items():
        x, y = map(int, key.split(','))
        distribution_dict[(x, y)] = (value['uids'], value['weights'])

    return distribution_dict

def yuma_example(K: float, E : float, distribution : dict[tuple[int, int], tuple[list[int], list[int]]]):
    active_stake = [x[1] for x in distribution.keys()]

    weights = [list(zip(x, y)) for x, y in distribution.values()]
    uids = [x[0] for x in distribution.keys()]
    scoring = dict(zip(uids, weights))
    columns = len(scoring)
    weight_sprase = inplace_row_normalize_sparse(get_weights_sparse(scoring))

    active_stake = inplace_normalize(active_stake)
    # Server prerank
    # preranks = matmul_sparse(weight_sprase, active_stake, columns)
    # Server consensus weight

    consensus = weighted_median_col_sparse(active_stake, weight_sprase, columns, K)
    # Consensus-clipped weight
    weight_sprase = col_clip_sparse(weight_sprase, consensus)

    plot_consensus(E, weight_sprase, active_stake)

if __name__ == "__main__":
    # Constants 
    # emission ration
    E = 0.5  #  half to miners and validators
    #  kappa, the majority weight
    K = 0.5
    distribution = validator_settings()
    
    for key, (list1, list2) in distribution.items():
        assert len(list1) == len(list2), f"Lists for key {key} are not the same size"
    
    expected_uids = set(range(len(distribution)))
    actual_uids = set(uid for uid, _ in distribution.keys())

    assert actual_uids == expected_uids, "UIDs are not in ascending order"

    yuma_example(
        K=K,
        E=E,
        distribution=distribution
    )


