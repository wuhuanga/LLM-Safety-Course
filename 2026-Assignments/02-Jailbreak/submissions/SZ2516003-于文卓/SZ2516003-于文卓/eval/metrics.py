def compute_asr(results):

    success = sum(r["attack_success"] for r in results)

    return success / len(results)


def compute_fpr(results):

    benign = [r for r in results if r["is_benign"]]

    if len(benign) == 0:
        return 0

    blocked = sum(r["blocked"] for r in benign)

    return blocked / len(benign)


def compute_avg_latency(results):

    total = sum(r["latency"] for r in results)

    return total / len(results)